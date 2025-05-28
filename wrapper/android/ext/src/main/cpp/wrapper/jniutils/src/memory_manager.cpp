#include "memory_manager.h"

MemoryManager::MemoryManager(std::size_t blockSize, std::size_t blockCount) : userBlockSize_(blockSize),
                                                                              totalBlockSize_(
                                                                                      sizeof(PoolHeader) +
                                                                                      blockSize),
                                                                              blockCount_(blockCount) {
    data_.resize(totalBlockSize_ * blockCount_);

    freeListHead_ = data_.data();
    unsigned char *ptr = data_.data();
    for (std::size_t i = 0; i < blockCount_ - 1; ++i) {
        *reinterpret_cast<unsigned char **>(ptr) = ptr + totalBlockSize_;
        ptr += totalBlockSize_;
    }
    *reinterpret_cast<unsigned char **>(ptr) = nullptr;
}

template<typename T, typename... Args>
std::unique_ptr<T, void (*)(T*)> MemoryManager::create(Args&&... args) {
    static_assert(sizeof(T) <= userBlockSize_, "Object doesn't fit in the pool's block size!");

    void *block = allocateBlock();
    if (!block) return {nullptr, nullptr};

    unsigned char *userPtr = static_cast<unsigned char *>(block) + sizeof(PoolHeader);

    T *objPtr = new(userPtr) T(std::forward<Args>(args)...);

    return std::unique_ptr<T, void (*)(T*)>(objPtr, &MemoryManager::deleter<T>);
}

template<typename T>
void MemoryManager::deleter(T *obj) {
    if (!obj) return;

    obj->~T();

    unsigned char *blockStart = reinterpret_cast<unsigned char *>(obj) - sizeof(PoolHeader);

    auto header = reinterpret_cast<PoolHeader *>(blockStart);
    MemoryManager *pool = header->pool;

    pool->deallocateBlock(blockStart);
}

void *MemoryManager::allocateBlock() {
    if (!freeListHead_) {
        return nullptr;
    }
    unsigned char *block = freeListHead_;
    freeListHead_ = *reinterpret_cast<unsigned char **>(block);

    auto header = reinterpret_cast<PoolHeader *>(block);
    header->pool = this;

    return block;
}

void MemoryManager::deallocateBlock(void *ptr) {
    auto *block = static_cast<unsigned char *>(ptr);
    *reinterpret_cast<unsigned char **>(block) = freeListHead_;
    freeListHead_ = block;
}
