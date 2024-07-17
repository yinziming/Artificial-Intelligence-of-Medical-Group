package com.example.strokereasoning.controller;

import com.example.strokereasoning.Service.UserService;
import com.example.strokereasoning.domain.Person;
import com.example.strokereasoning.domain.Result;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;


@RestController
@RequestMapping("/user")
@CrossOrigin
public class UserController {

    private UserService userService;
    @Autowired
    public void setUserService(UserService userService) {
        this.userService = userService;
    }

    /**
     * 根据id删除用户
     * @param name 用户name
     * @return result
     */
    @PostMapping("deleteUser")
    public @ResponseBody
    Result<String> deleteUser(String name){
        if(userService.deleteUser(name)){
            return new Result<String>(true,200,"删除成功");
        }else{
            return new Result<String>(false,500,"删除失败");
        }
    }

    /**
     * 根据用户名查询用户
     * @param name 用户名
     * @return 病人列表
     */
    @GetMapping("showUserInfo")
    public Result<Person> findUser(String name){
        System.out.println("name"+name);
        Person user = userService.findUser(name);
        if (user != null){
            return new Result<Person>(true,200,"查询成功", user);
        }else{
            return new Result<Person>(false,500,"查询失败");
        }
    }

}