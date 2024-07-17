package com.example.strokereasoning.Service.impl;

import com.example.strokereasoning.Service.Performdrools;
import com.example.strokereasoning.domain.Person;
import com.example.strokereasoning.domain.droolsinsert;
import org.springframework.stereotype.Service;

@Service
public class Performdroolsimpl implements Performdrools {

    @Override
    public String[] ptarray(Person person){
        droolsinsert function = new droolsinsert();
        function.get(person);


        return person.getArrayPt();

    }
    @Override
    public String[] otarray(Person person){
        droolsinsert function = new droolsinsert();
        function.get(person);
        return person.getArrayOt();

    }
    @Override
    public String[] starray(Person person){
        droolsinsert function = new droolsinsert();
        function.get(person);
        return person.getArraySt();

    }





    /*public static void main(String[] args) {

        // 1. 创建KieServices实例并加载KieContainer
        KieServices kieServices = KieServices.Factory.get();

        KieContainer kieContainer = kieServices.getKieClasspathContainer();
//       //        KieContainer kieContainer = kieServices.newKieClasspathContainer("CheckAdult"); KieBase kBase = kieContainer.getKieBase("CheckAdult");

        // 2. 创建KieSession   "AdultKS"
        KieSession kieSession = kieContainer.newKieSession("AdultKS");
        //这一段代码里kiesession就是null

        // 3. 设置事实对象"John",25
        demoperson.Person person = new demoperson.Person("John",36,3,41, 1,
                1,2,91,3,2,80,30,130,0.92,10,
                8,50,0.5,39);
        System.out.println(person.name);
        System.out.println("<-----下面内容为规则推理内容----->");

        kieSession.insert(person);

        // 4. 触发规则执行
        kieSession.fireAllRules();

        // 5. 关闭KieSession
        kieSession.dispose();
    }
*/






}
