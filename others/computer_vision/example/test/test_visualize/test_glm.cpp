/**
 * test glm
 * @author: xiaotaw
 * @email: 
 * @date: 2020/08/24 12:47
 */
#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp> // for glm::to_string

int main()
{
    // 测试列主序
    glm::mat4 trans = glm::translate(glm::mat4(1.0f), glm::vec3(1.0f, 1.0f, 0.0f));
    std::cout << glm::to_string(trans) << std::endl;
    glm::vec4 res =  trans * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); 
    std::cout << glm::to_string(res) << std::endl;
    return 0;
}