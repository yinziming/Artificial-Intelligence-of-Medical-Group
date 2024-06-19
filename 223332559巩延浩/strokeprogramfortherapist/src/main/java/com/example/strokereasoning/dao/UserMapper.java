package com.example.strokereasoning.dao;
import com.example.strokereasoning.domain.Person;
import org.apache.ibatis.annotations.Param;

public interface UserMapper {

    /**
     * 查询用户
     *
     * @param name 用户名
     * @return 返回查询到的用户信息
     */
    Person findUser(@Param("name") String name);

    /**
     * 根据name删除用户
     *
     * @param name 用户name
     * @return
     */
    boolean deleteUser(@Param("name") String name);



}
