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
}
