package com.example.strokereasoning.domain;

import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;

public class droolsinsert{
//    public Person obj;

    public static void get(Person obj){


            // 1. 创建KieServices实例并加载KieContainer
        KieServices kieServices = KieServices.Factory.get();
        KieContainer kieContainer = kieServices.getKieClasspathContainer();
        KieSession kieSession = kieContainer.newKieSession("Rehabilitation");
            // 3. 设置事实对象"John",25
            // com.example.strokereasoning.domain.Person person = new com.example.strokereasoning.domain.Person("John",36,3,41, 1,1,2,91,3,2,80,30,130,0.92,10,8,50,0.5,39);
            //    检查是否有禁忌症。
//        Person person2 =obj;
        obj.contraindication();
        System.out.println(obj.getName());
        System.out.println("<-----下面内容为规则推理内容----->");

        kieSession.insert(obj);
            // 4. 触发规则执行
        kieSession.fireAllRules();
            //测试返回的字符串是否可以输出。
        System.out.println("<-----下面内容为getArraypt----->");
//        System.out.println(obj.getArraypt()[0]);
            // 5. 关闭KieSession
        kieSession.dispose();


    }
}
