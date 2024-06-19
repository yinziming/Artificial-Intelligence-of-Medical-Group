package com.example.strokereasoning;

import org.mybatis.spring.annotation.MapperScan;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.web.bind.annotation.CrossOrigin;

@SpringBootApplication()
@CrossOrigin
@MapperScan("com.example.strokereasoning.dao")
public class RecoveryApplication {

    public static void main(String[] args) {
        SpringApplication.run(com.example.strokereasoning.RecoveryApplication.class, args);
        System.out.println("你好");


    }
}

/*KieServices kieServices = KieServices.Factory.get();

        KieContainer kieContainer = kieServices.getKieClasspathContainer();
        //        KieContainer kieContainer = kieServices.newKieClasspathContainer("CheckAdult"); KieBase kBase = kieContainer.getKieBase("CheckAdult");

        // 2. 创建KieSession   "AdultKS"
        KieSession kieSession = kieContainer.newKieSession("Rehabilitation");


        // 3. 设置事实对象"John",25
        com.example.strokereasoning.domain.Person person = new com.example.strokereasoning.domain.Person("John",36,3,41, 1,
                1,2,91,3,2,80,30,130,0.92,10,
                8,50,0.5,39);



        //    检查是否有禁忌症。
        person.contraindication();
        System.out.println(person.name);

        System.out.println("<-----下面内容为规则推理内容----->");

        kieSession.insert(person);

        // 4. 触发规则执行
        kieSession.fireAllRules();
        //测试返回的字符串是否可以输出。
        System.out.println("<-----下面内容为getArraypt----->");
        System.out.println(person.getArraypt()[0]);

        // 5. 关闭KieSession
        kieSession.dispose();*/