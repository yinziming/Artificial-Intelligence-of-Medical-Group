package com.example.strokereasoning.Service.impl;

import com.example.strokereasoning.Service.Persondata;
import com.example.strokereasoning.domain.Person;
import org.springframework.stereotype.Service;


@Service
public class Persondataimpl implements Persondata {
    @Override
    public Person GetData(String name, int age, int S5q, int MrcSum, int BBS_Sit_to_stand,
                          int BBS_Standing, int BBS_Siting, int Mmasa, int FoisGrade, int Rass,
                          int HR, int RR, int SBP,int DBP, double SPO2dFiO2, int ICP, int PEEP, double FiO2,
                          int T) {


        Person person = new Person( name,age,S5q,MrcSum,BBS_Sit_to_stand,BBS_Standing,BBS_Siting,Mmasa,FoisGrade,
                                    Rass,HR, RR,  SBP, DBP, SPO2dFiO2,  ICP,  PEEP,  FiO2,T);

        System.out.println("接收到的数据："+ person);
        return person;
    }

}


/*

//        接收数据  String name, int age, int s5q, int MRCsum, int BBS_Sit_to_stand,
//                          int BBS_Standing, int BBS_Siting, int MMASA, int FOISgrade, int RASS,
//                          int HR, int RR, int SBP, double SPO2, int ICP, int PEEP, int MAP, double FiO2,
//                          int T



public String name;
    public int age;
    public boolean isAdult;
    //    下面一行增加自己写的康复评估指标：s5q、MRCsum、BBS_Sit_to_stand、BBS_Stand、BBS_Siting、MMASA、FOISgrade.
    public int s5q,MRCsum,BBS_Sit_to_stand,BBS_Standing,BBS_Siting,MMASA,FOISgrade,RASS;
    //    下面一行增加自己写的康复评估指标:HR：心率,RR：呼吸频率,SBP：收缩压,SPO2：血氧饱和度,ICP：颅内压,PEEP：呼气末正压,MAP：平均动脉压,FiO2：吸氧分数,T：体温;
    public int HR,RR,SBP,ICP,PEEP,MAP,T;
    public double SPO2,FiO2;


    public Person(String name, int age,int s5q,int MRCsum,int BBS_Sit_to_stand,
                  int BBS_Standing,int BBS_Siting,int MMASA,int FOISgrade,int RASS,
                  int HR,int RR,int SBP,double SPO2,int ICP,int PEEP,int MAP,double FiO2,
                  int T)
    {
        this.name = name;
        this.age = age;
        this.s5q = s5q;this.RASS = RASS;

        this.MRCsum = MRCsum;
        this.BBS_Sit_to_stand = BBS_Sit_to_stand;
        this.BBS_Standing = BBS_Standing;
        this.BBS_Siting = BBS_Siting;
        this.MMASA = MMASA;
        this.FOISgrade = FOISgrade;

        this.HR = HR;
        this.RR = RR;
        this.SBP = SBP;
        this.SPO2 = SPO2;
        this.ICP = ICP;
        this.PEEP = PEEP;
        this.MAP = MAP;
        this.FiO2 = FiO2;
        this.T = T;
    }


    // Getters and setters
    public void contraindication() {
        if (HR>120 || HR<40 || RR>35 || SBP<90 || SBP>180 || SPO2<=0.9 || ICP>=15 || PEEP>=10 || MAP<60
                || FiO2>=0.6 || T>40 )
        {
            System.out.println("该患者目前生命体征不适合做康复，建议24h后重新评估");
            System.out.println("<----!!----->");

        }
        System.out.println("请仔细检查病人是否有不稳定性骨折、急性手术...等康复训练禁忌症");


    }

    public String getName() {
        return name;
    }


    public void setName(String name) {
        this.name = name;
    }

    public int getAge() {
        return age;
    }

    public void setAge(int age) {
        this.age = age;
    }

    public boolean isAdult() {
        return isAdult;
    }

    public void setAdult(boolean adult) {
        isAdult = adult;
    }
}
*/