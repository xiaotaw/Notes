
#define DISABLE_COPY_ASSIGN_MOVE(T)   \
    T(const T &) = delete;            \
    T &operator=(const T &) = delete; \
    T(T &&) = delete;                 \
    T &operator=(T &&) = delete


#define DISABLE_COPY_ASSIGN(T)   \
    T(const T &) = delete;            \
    T &operator=(const T &) = delete

