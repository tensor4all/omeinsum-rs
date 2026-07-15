use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;

#[derive(Default)]
pub(crate) struct ScratchPool<T> {
    free: Vec<Vec<T>>,
}

pub(crate) struct ScratchBuffer<'a, T> {
    buf: Vec<T>,
    pool: &'a mut ScratchPool<T>,
}

impl<T: Default> ScratchPool<T> {
    pub(crate) fn acquire(&mut self, len: usize) -> ScratchBuffer<'_, T> {
        let reuse_index = self
            .free
            .iter()
            .enumerate()
            .filter(|(_, buf)| buf.capacity() >= len)
            .min_by_key(|(_, buf)| buf.capacity())
            .map(|(index, _)| index);
        let mut buf = reuse_index
            .map(|index| self.free.swap_remove(index))
            .unwrap_or_else(|| Vec::with_capacity(len));
        buf.clear();
        buf.resize_with(len, T::default);

        ScratchBuffer { buf, pool: self }
    }
}

impl<T> ScratchBuffer<'_, T> {
    #[cfg(test)]
    pub(crate) fn as_mut_slice(&mut self) -> &mut [T] {
        self.buf.as_mut_slice()
    }

    pub(crate) fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }

    #[cfg(test)]
    pub(crate) fn capacity(&self) -> usize {
        self.buf.capacity()
    }
}

impl<T> Drop for ScratchBuffer<'_, T> {
    fn drop(&mut self) {
        let mut buf = std::mem::take(&mut self.buf);
        buf.clear();
        self.pool.free.push(buf);
    }
}

thread_local! {
    static PACKING_BUFFERS: RefCell<HashMap<TypeId, Box<dyn Any>>> =
        RefCell::new(HashMap::new());
}

const MAX_PACKING_BUFFERS_PER_TYPE: usize = 2;
const MAX_PACKING_BUFFER_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct PackingBuffer<T: Copy + 'static> {
    buf: Vec<T>,
    pooled: bool,
}

impl<T: Copy + Default + 'static> PackingBuffer<T> {
    pub(crate) fn acquire(len: usize) -> Self {
        let bytes = len.saturating_mul(std::mem::size_of::<T>());
        if bytes == 0 || bytes > MAX_PACKING_BUFFER_BYTES {
            return Self {
                buf: vec![T::default(); len],
                pooled: false,
            };
        }

        let mut buf = PACKING_BUFFERS.with(|pool| {
            let mut pool = pool.borrow_mut();
            let buffers = pool
                .entry(TypeId::of::<T>())
                .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()))
                .downcast_mut::<Vec<Vec<T>>>()
                .expect("packing buffer pool type mismatch");
            let best = buffers
                .iter()
                .enumerate()
                .filter(|(_, buffer)| buffer.capacity() >= len)
                .min_by_key(|(_, buffer)| buffer.capacity())
                .map(|(index, _)| index);
            best.map(|index| buffers.swap_remove(index))
                .unwrap_or_else(|| Vec::with_capacity(len))
        });
        buf.truncate(len);
        buf.resize_with(len, T::default);

        Self { buf, pooled: true }
    }

    pub(crate) fn as_slice(&self) -> &[T] {
        &self.buf
    }

    pub(crate) fn as_mut_vec(&mut self) -> &mut Vec<T> {
        &mut self.buf
    }
}

impl<T: Copy + 'static> Drop for PackingBuffer<T> {
    fn drop(&mut self) {
        if !self.pooled {
            return;
        }

        let buf = std::mem::take(&mut self.buf);
        let bytes = buf.capacity().saturating_mul(std::mem::size_of::<T>());
        if bytes == 0 || bytes > MAX_PACKING_BUFFER_BYTES {
            return;
        }

        let _ = PACKING_BUFFERS.try_with(|pool| {
            let Ok(mut pool) = pool.try_borrow_mut() else {
                return;
            };
            let buffers = pool
                .entry(TypeId::of::<T>())
                .or_insert_with(|| Box::new(Vec::<Vec<T>>::new()))
                .downcast_mut::<Vec<Vec<T>>>()
                .expect("packing buffer pool type mismatch");
            if buffers.len() < MAX_PACKING_BUFFERS_PER_TYPE {
                buffers.push(buf);
            } else if let Some((smallest, capacity)) = buffers
                .iter()
                .enumerate()
                .map(|(index, buffer)| (index, buffer.capacity()))
                .min_by_key(|(_, capacity)| *capacity)
            {
                if capacity < buf.capacity() {
                    buffers[smallest] = buf;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scratch_pool_reuses_released_capacity() {
        let mut pool = ScratchPool::<f32>::default();
        let mut first = pool.acquire(32);
        first.as_mut_slice().fill(1.0);
        drop(first);

        let second = pool.acquire(16);
        assert!(second.capacity() >= 32);
    }

    #[test]
    fn test_scratch_pool_grows_when_requested_capacity_is_larger() {
        let mut pool = ScratchPool::<f32>::default();
        let small = pool.acquire(8);
        drop(small);

        let large = pool.acquire(128);
        assert!(large.capacity() >= 128);
    }

    #[test]
    fn test_packing_buffer_reuses_initialized_storage() {
        #[derive(Clone, Copy, Default)]
        struct Marker(u32);

        let mut first = PackingBuffer::<Marker>::acquire(127);
        first.as_mut_vec().fill(Marker(42));
        let pointer = first.as_slice().as_ptr();
        drop(first);

        let second = PackingBuffer::<Marker>::acquire(127);
        assert_eq!(second.as_slice().as_ptr(), pointer);
        assert!(second.as_slice().iter().all(|value| value.0 == 42));
    }

    #[test]
    fn test_packing_buffer_keeps_two_largest_and_uses_best_fit() {
        #[derive(Clone, Copy, Default)]
        struct Marker(u8);

        let smallest = PackingBuffer::<Marker>::acquire(8);
        let smallest_capacity = smallest.buf.capacity();
        drop(smallest);

        let middle = PackingBuffer::<Marker>::acquire(smallest_capacity + 1);
        let middle_capacity = middle.buf.capacity();
        drop(middle);

        let largest = PackingBuffer::<Marker>::acquire(middle_capacity + 1);
        let largest_capacity = largest.buf.capacity();
        drop(largest);

        PACKING_BUFFERS.with(|pool| {
            let pool = pool.borrow();
            let buffers = pool
                .get(&TypeId::of::<Marker>())
                .unwrap()
                .downcast_ref::<Vec<Vec<Marker>>>()
                .unwrap();
            let mut capacities: Vec<_> = buffers.iter().map(Vec::capacity).collect();
            capacities.sort_unstable();
            assert_eq!(capacities, vec![middle_capacity, largest_capacity]);
        });

        let best_fit = PackingBuffer::<Marker>::acquire(middle_capacity);
        assert_eq!(best_fit.buf.capacity(), middle_capacity);
        assert!(best_fit.as_slice().iter().all(|value| value.0 == 0));
    }

    #[test]
    fn test_packing_buffer_drop_tolerates_reentrant_pool_borrow() {
        #[derive(Clone, Copy, Default)]
        struct Marker(u8);

        let buffer = PackingBuffer::<Marker>::acquire(8);
        assert!(buffer.as_slice().iter().all(|value| value.0 == 0));
        PACKING_BUFFERS.with(|pool| {
            let pool = pool.borrow_mut();
            drop(buffer);
            let buffers = pool
                .get(&TypeId::of::<Marker>())
                .unwrap()
                .downcast_ref::<Vec<Vec<Marker>>>()
                .unwrap();
            assert!(buffers.is_empty());
        });
    }

    #[test]
    fn test_packing_buffer_does_not_retain_zero_byte_storage() {
        #[derive(Clone, Copy, Default)]
        struct Marker;

        let buffer = PackingBuffer {
            buf: vec![Marker; 8],
            pooled: true,
        };
        drop(buffer);

        PACKING_BUFFERS.with(|pool| {
            assert!(!pool.borrow().contains_key(&TypeId::of::<Marker>()));
        });
    }
}
