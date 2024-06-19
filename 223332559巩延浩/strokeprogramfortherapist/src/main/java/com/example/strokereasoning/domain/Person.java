package com.example.strokereasoning.domain;

import java.util.Arrays;

//需要增加脑卒中患者的相关指标。
public class Person {
    private String name;
    private final int age;

    //    下面一行增加自己写的康复评估指标：s5q、MRCsum、BBS_Sit_to_stand、BBS_Stand、BBS_Siting、MMASA、FOISgrade.
    private final int S5q;
    private final int MrcSum;
    private final int BBS_Standing;
    private final int BBS_Siting;
    private final int BBS_Sit_to_stand;
    private final int Rass;
    private final int Mmasa;
    private final int FoisGrade;
    //    下面一行增加自己写的康复评估指标:HR：心率,RR：呼吸频率,SBP：收缩压,
    //    double : SPO2：血氧饱和度,ICP：颅内压,PEEP：呼气末正压,MAP：平均动脉压,FiO2：吸氧分数,T：体温;
    private final double HR;
    private final double RR;
    private final double SBP;
    private final double DBP;
    private final double ICP;
    private final double PEEP;
    private double MAP;
    private final double T;
    private final double SPO2dFiO2;
    private final double FiO2;

    private String[] ArrayPt;
    private String[] ArrayOt;
    private String[] ArraySt;

    private String[] ArrayPtAdvice;
    private String[] ArrayOtAdvice;
    private String[] ArrayStAdvice;

    //int MAP
    public Person(String name, int age, int S5q, int MrcSum, int BBS_Sit_to_stand,
                  int BBS_Standing, int BBS_Siting, int Mmasa, int FoisGrade, int Rass,
                  double HR, double RR, double SBP, double DBP, double SPO2dFiO2, double ICP, double PEEP, double FiO2,
                  double T) {
        this.name = name;
        this.age = age;
        this.S5q = S5q;
        this.MrcSum = MrcSum;
        this.BBS_Standing = BBS_Standing;
        this.BBS_Sit_to_stand = BBS_Sit_to_stand;
        this.BBS_Siting = BBS_Siting;
        this.Mmasa = Mmasa;
        this.Rass = Rass;
        this.FoisGrade = FoisGrade;
        //此处对输入的数据做判断
        if(HR > 0)
            this.HR = HR;
        else
            this.HR = 85;

        if(RR > 0)
            this.RR = RR;
        else
            this.RR = 30;

        if(SBP > 0)
            this.SBP = SBP;
        else
            this.SBP = 100;
        if(DBP > 0)
            this.DBP = DBP;
        else
            this.DBP = 80;

        if(SPO2dFiO2 > 0)
            this.SPO2dFiO2 = SPO2dFiO2;
        else
            this.SPO2dFiO2 = 0.92;

        if(ICP > 0)
            this.ICP = ICP;
        else
            this.ICP = 10;

        if(PEEP > 0)
            this.PEEP = PEEP;
        else
            this.PEEP = 10;
//        this.MAP = MAP;

        if(FiO2 > 0)
            this.FiO2 = FiO2;
        else
            this.FiO2 = 0.5;

        if(T > 0)
            this.T = T;
        else
            this.T = 37;
    }

    @Override
    public String toString() {
        return "Person{" +
                "name='" + name + '\'' +
                ", age=" + age +
                ", S5q=" + S5q +
                ", MrcSum=" + MrcSum +
                ", BBS_Standing=" + BBS_Standing +
                ", BBS_Siting=" + BBS_Siting +
                ", BBS_Sit_to_stand=" + BBS_Sit_to_stand +
                ", Rass=" + Rass +
                ", Mmasa=" + Mmasa +
                ", FoisGrade=" + FoisGrade +
                ", HR=" + HR +
                ", RR=" + RR +
                ", SBP=" + SBP +
                ", DBP=" + DBP +
                ", ICP=" + ICP +
                ", PEEP=" + PEEP +
//                ", MAP=" + MAP +
                ", T=" + T +
                ", SPO2=" + SPO2dFiO2 +
                ", FiO2=" + FiO2 +
                ", ArrayPt=" + Arrays.toString(ArrayPt) +
                ", ArrayOt=" + Arrays.toString(ArrayOt) +
                ", ArraySt=" + Arrays.toString(ArraySt) +
                ", ArrayPt=" + Arrays.toString(ArrayPtAdvice) +
                ", ArrayOt=" + Arrays.toString(ArrayOtAdvice) +
                ", ArraySt=" + Arrays.toString(ArrayStAdvice) +
                '}';
    }

    // Getters and setters

    /**
     * 有禁忌症 返回true   没有返回false
     *
     * @return
     */
    public boolean contraindication() {
//        if (HR>120 || HR<40 || RR>35 || SBP<=90 || SBP>=180 || SPO2<=0.9 || ICP>=15 || PEEP>=10 || MAP<60
//                || FiO2>=0.6 || T>40 ) {
//           System.out.println("该患者目前生命体征不适合做康复，建议24h后重新评估");
//           System.out.println("<----!!----->");
//        }
//        System.out.println("请仔细检查病人是否有不稳定性骨折、急性手术...等康复训练禁忌症");
        MAP = (SBP + 2 * DBP) / 3;
        System.out.println("map="+MAP);
        System.out.println("sbp="+SBP);
        System.out.println("<----!jisuan map!----->");
        //该处判断生命体征值是否为空，若为空，则置一个默认值
        /* if(HR == null) HR = 80;
        if(RR == null) RR = 30;
        if(SBP == null) SBP = 100;
        if(SPO2dFiO2 == null) SPO2dFiO2 = 1;
        if(ICP == null) ICP = 12;
        if(PEEP == null) PEEP = 9;
        if(FiO2 == null) FiO2 = 0.6;
        if(T == null) T = 38;
        return HR > 120 || HR < 40 || RR > 35 || SBP < 90 || SBP > 180 || SPO2dFiO2 <= 0.9 || ICP >= 15 || PEEP >= 10 || MAP < 60
                || FiO2 >= 0.6 || T > 40;
        */


        return HR > 120 || HR < 40 || RR > 35 || SBP < 90 || SBP > 180 || SPO2dFiO2 <= 0.9 || ICP >= 15 || PEEP >= 10 || MAP < 60
                || FiO2 >= 0.6 || T > 40;
    }

    public String[] getArrayPt() {
        return ArrayPt;
    }

    public void setArrayPt(String[] ArrayPt) {
        this.ArrayPt = ArrayPt;
    }

    public String[] getArrayOt() {
        return ArrayOt;
    }

    public void setArrayOt(String[] ArrayOt) {
        this.ArrayOt = ArrayOt;
    }

    public String[] getArraySt() {
        return ArraySt;
    }

    public void setArraySt(String[] ArraySt) {
        this.ArraySt = ArraySt;
    }

    //    关于医嘱项目advice
    public String[] getArrayPtAdvice() {
        return ArrayPtAdvice;
    }

    public void setArrayPtAdvice(String[] ArrayPtAdvice) {
        this.ArrayPtAdvice = ArrayPtAdvice;
    }

    public String[] getArrayOtAdvice() {
        return ArrayOtAdvice;
    }

    public void setArrayOtAdvice(String[] ArrayOtAdvice) {
        this.ArrayOtAdvice = ArrayOtAdvice;
    }

    public String[] getArrayStAdvice() {
        return ArrayStAdvice;
    }

    public void setArrayStAdvice(String[] ArrayStAdvice) {
        this.ArrayStAdvice = ArrayStAdvice;
    }

    /*
    String name, int age, int S5q, int MrcSum, int BBS_Sit_to_stand,
    int BBS_Standing, int BBS_Siting, int Mmasa, int FoisGrade, int Rass,
    int HR, int RR, int SBP,int DBP, double SPO2dFiO2, int ICP, int PEEP, double FiO2,
    int T
    */


    public void setName(String name) {
        this.name = name;}
        public String getName() {
        return name;
    }

    /*    康复评估所有的指标：public int getS5q() { return s5q; }
     */
    public int getBBS_Siting() { return BBS_Siting; }
    public int getAge() { return age; }
    public int getBBS_Standing() { return BBS_Standing; }
    public int getFoisGrade() { return FoisGrade; }
    public int getBBS_Sit_to_stand() { return BBS_Sit_to_stand; }
    public int getMmasa() { return Mmasa; }
    public int getRass() { return Rass; }
    public int getMrcSum() { return MrcSum; }
    public int getS5q() { return S5q; }

}




