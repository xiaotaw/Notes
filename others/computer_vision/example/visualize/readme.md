## reference
https://learnopengl-cn.github.io/

## glad 
glad comes from https://glad.dav1d.de/

## Notes:
1. in glad.c, line 25, modify '#include <glad/glad.h>' into '#include "glad/glad.h"'
2. if error "undefined reference to symbol 'dlclose@@GLIBC_2.2.5'" occurs, add dl library. see in visualize/CMakeLists.txt