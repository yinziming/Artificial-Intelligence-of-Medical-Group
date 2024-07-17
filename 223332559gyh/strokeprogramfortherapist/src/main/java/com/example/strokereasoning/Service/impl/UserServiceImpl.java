package com.example.strokereasoning.Service.impl;

import com.example.strokereasoning.Service.UserService;
import com.example.strokereasoning.dao.UserMapper;
import com.example.strokereasoning.domain.Person;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

/**
 * UserService的实现类
 */
@Service
public class UserServiceImpl implements UserService {

    private UserMapper userMapper;

    /**
     * 注入dao
     * @param userMapper userMapper
     */
    @Resource
    public void setUserMapper(UserMapper userMapper){
        this.userMapper = userMapper;
    }


    /**
     * 登录成功查询用户信息
     * @param name 用户名
     * @return 用户所有信息
     */
    @Override
    public Person findUser(String name) {
        return userMapper.findUser(name);
    }

    /**
     * 删除用户信息
     * @param name 用户name
     * @return 受影响记录条数
     */
    @Override
    public boolean deleteUser(String name) {
        return userMapper.deleteUser(name);
    }


    }

