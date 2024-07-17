package com.example.strokereasoning.Service;
import com.example.strokereasoning.domain.Person;

public interface UserService {

    /**
     * 查询用户信息
     * @param name 用户名
     * @return 用户信息
     */
    Person findUser(String name);


    /**
     * 删除用户信息
     * @param name 根据用户name删除用户信息
     * @return 受影响记录条数
     */
   boolean deleteUser(String name);

}
