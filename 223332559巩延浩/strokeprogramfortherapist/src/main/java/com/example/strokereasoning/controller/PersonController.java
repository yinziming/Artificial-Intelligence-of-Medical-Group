package com.example.strokereasoning.controller;

import com.example.strokereasoning.domain.Person;
import com.example.strokereasoning.domain.Result;
import com.example.strokereasoning.domain.ResultData;
import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;


@RestController
@RequestMapping
public class PersonController {
    //getSolution
    @GetMapping("/person")
//    public Result<ResultData> AcceptData(@RequestBody String name, int age, int s5q, int MrcSum, int BBS_Sit_to_stand,
//                                         int BBS_Standing, int BBS_Siting, int Mmasa, int FoisGrade, int Rass,
//                                         double HR, double RR, double SBP, double SPO2, double ICP, double PEEP, double MAP, double FiO2,
//                                         double T){
    public Result<ResultData> AcceptData(){
        System.out.println("/person开始：");
        Person person007 = new Person("John",36,4,41, 1,1,1,96,3,3,80,30,130,80,10,8,50,0.5,39);
//        Person person007 = new Person( name,  age,  s5q,  MrcSum,  BBS_Sit_to_stand,BBS_Standing,
//                                        BBS_Siting,  Mmasa,  FoisGrade,  Rass,HR,  RR,  SBP,  SPO2,
//                                        ICP,  PEEP,  MAP,  FiO2,T);

        person007.contraindication();   //    检查是否有禁忌症。
        System.out.println("<-----下面内容为规则推理内容27----->");
        KieServices kieServices = KieServices.Factory.get();
        KieContainer kieContainer = kieServices.getKieClasspathContainer();
        KieSession kieSession = kieContainer.newKieSession("Rehabilitation");   // 2. 创建KieSession
        // 3. 设置事实对象"John",25

        //    检查是否有禁忌症。  Person person007
        person007.contraindication();

        System.out.println("<-----下面内容为规则推理内容36----->");

        kieSession.insert(person007);
        kieSession.fireAllRules();
        System.out.println("<-----下面内容为getArrayPt----->");

        //默认输出


        String[] arrayOt = person007.getArrayOt();
        String[] arraySt = person007.getArraySt();
        String[] arrayPt = person007.getArrayPt();

        System.out.println(person007.getArrayOt()[0]);
        System.out.println();
        kieSession.dispose();
//        DroolsInsert.get(person007);

        ResultData resultdata = new ResultData(person007.getArrayPt(),person007.getArrayOt(),person007.getArraySt());
        Result<ResultData> result = new Result(true,200,"success",resultdata);
        System.out.println("result.getData():");
        System.out.println(result.getData());

        return  result;
//        return jsonObject;

    }
}



/* @PostMapping("/data")
    public ResponseEntity<String> receivePersonData(@RequestBody Person person) {
        // 在这里处理接收到的Person对象数据
        // 可以通过person.getName()、person.getAge()、person.getBirthDate()获取数据
        person.getName();
        person.getAge();
        person.gets5q();
        person.getBBS_Sit_to_stand();
        return ResponseEntity.ok("Data received successfully");
    }

    KieServices kieServices = KieServices.Factory.get();
        KieContainer kieContainer = kieServices.getKieClasspathContainer();
        KieSession kieSession = kieContainer.newKieSession("Rehabilitation");   // 2. 创建KieSession
        // 3. 设置事实对象"John",25
        Person person007 = new Person("John",36,3,41, 1,1,2,91,3,2,80,30,130,0.92,10,8,50,0.5,39);
        //    检查是否有禁忌症。
        person007.contraindication();
        System.out.println(person007.getName());
        System.out.println("<-----下面内容为规则推理内容----->");

        kieSession.insert(person007);
        kieSession.fireAllRules();
        System.out.println("<-----下面内容为getArraypt----->");
        System.out.println(person007.getArrayst()[0]);
        kieSession.dispose();*/
