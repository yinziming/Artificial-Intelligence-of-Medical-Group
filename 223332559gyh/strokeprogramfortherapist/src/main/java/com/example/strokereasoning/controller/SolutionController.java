package com.example.strokereasoning.controller;


import com.alibaba.fastjson.JSON;
import com.alibaba.fastjson.serializer.SerializerFeature;
import com.example.strokereasoning.domain.Person;
import org.kie.api.KieServices;
import org.kie.api.runtime.KieContainer;
import org.kie.api.runtime.KieSession;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RestController;

import java.util.LinkedHashMap;
import java.util.Map;

@RestController
public class SolutionController {
    @PostMapping("/getSolution")
    public String getSolution(@RequestBody Person person) {
        System.out.println(1);
        System.out.println(person.toString());
        Map<String, Object> result = new LinkedHashMap<String, Object>();
        if (person.contraindication()) {
            // 有禁忌症
            result.put("contraindication", "该患者目前生命体征不适合做康复，建议24h后重新评估");
        } else {
            result.put("contraindication", null);
        }

        /*判断是否为空值  if(HR == null) HR = 80;
        if(person.getHR() == null) {
            person.setHR(80);
        }
        if(person.getRR() == null) {
            person.setRR(30);
        }
        if(person.getSBP() == null) {
            person.setSBP(100);
        }
        if(person.getSPO2dFiO2() == null) {
            person.setSPO2dFiO2(1);
        }
        if(person.getICP() == null) {
            person.setICP(14);
        }
        if(person.getHR() == null) {
            person.setHR(80);
        }
        if(person.getHR() == null) {
            person.setHR(80);
        }

         */
        System.out.println(1006);
        System.out.println(person.contraindication());
        System.out.println(250);
        result.put("tips1", "请仔细检查病人是否有不稳定性骨折、急性手术、精神不稳定、不稳定性心律失常、活动性出血、气道不安全等康复训练禁忌症");
        result.put("tips2", "注意！(1)电刺激禁忌症：局部金属，带有心脏起搏器，避开劲动脉窦;(2)气压和振动的禁忌症：下肢静脉血栓形成;(3)TMS/tDCS 禁忌症");
        System.out.println("请仔细检查病人是否有不稳定性骨折、急性手术、精神不稳定、不稳定性心律失常、活动性出血、气道不安全等康复训练禁忌症");
        System.out.println("注意！\n(1)电刺激禁忌症：局部金属，带有心脏起搏器，避开劲动脉窦;\n(2)气压和振动的禁忌症：下肢静脉血栓形成;\n(3)TMS/tDCS 禁忌症");
        KieServices kieServices = KieServices.Factory.get();
        KieContainer kieContainer = kieServices.getKieClasspathContainer();
        KieSession kieSession = kieContainer.newKieSession("Rehabilitation");   // 2. 创建KieSession
        kieSession.insert(person);
        kieSession.fireAllRules();

        String[] ptdefect={"主动床椅转移；坐于床边；辅助站立",
                "被动/主动关节活动；上下肢的抗阻训练；床上或坐位踏车；帮助下步行；神经肌肉电刺激；ADL指导"};
        String[] ptAdvicedefect={"PT治疗（偏瘫）床旁/ PT治疗（四肢瘫）床旁",
                "气压式血液循环驱动(器)(进口)",
                "下肢认知运动协调反馈训练",
                "生物反馈治疗（SWFK肌电生物反馈治疗）",
                "中频电疗床旁",
                "电动起立床训练",
                "康复踏车训练",
                "悬吊治疗",
                "局部电动按摩",
                "足踝机器人（单侧）",
                "疼痛治疗（GRD干扰电治疗）",
                "热湿疗法",
                "重复经颅磁刺激治疗/神经调节治疗（TDCS）"};
        String[] stdefect={"呼吸：体位训练，气道廓清技术，呼吸训练，咳嗽训练，运动训练，物理治疗",
                "吞咽：感觉刺激，口肌训练，辅助技巧，电刺激，呼吸训练，发声训练，构音训练，说话瓣膜",
                "失语：音乐治疗，自发语启动，口颜面肌肉训练，交流辅助装置，言语促通治疗，实用交流能力训练"};
        String[] stAdvicedefect={"ST治疗（吞咽床边）(运动再学习训练(MRP),吞咽障碍电刺激训练)",
                "ST治疗（失语床边/构音障碍床边）(言语矫正治疗,发声障碍训练)",
                "呼吸康复（肺功能综合训练）"};
        String[] otdefect={"多感觉刺激(音乐、嗅觉、浅感觉等)",
                "情绪支持(渐进放松、冥想、正念等)"};
        String[] otAdvicedefect={"OT治疗（床旁）",
                "OT治疗（认知训练加收）",
                "综合消肿治疗（单侧骨关节、偏瘫）"};
//        String[] arrayOt = person.getArrayOt();
//        String[] arraySt = person.getArraySt();
//        String[] arrayPt = person.getArrayPt();
        System.out.println(person.getArrayOt() == null);
        System.out.println("端午节");

        result.put("OT", person.getArrayOt() == null ? otdefect: person.getArrayOt());

        result.put("ST", person.getArraySt() == null ? stdefect : person.getArraySt());
        result.put("PT", person.getArrayPt() == null ? ptdefect : person.getArrayPt());
        result.put("OTAdvice", person.getArrayOtAdvice() == null ? otAdvicedefect : person.getArrayOtAdvice());
        result.put("STAdvice", person.getArrayStAdvice() == null ? stAdvicedefect : person.getArrayStAdvice());
        result.put("PTAdvice", person.getArrayPtAdvice() == null ? ptAdvicedefect: person.getArrayPtAdvice());
        System.out.println(person.getArrayPt());


        kieSession.dispose();


        return JSON.toJSONString(result, SerializerFeature.SortField);


    }

    @GetMapping("/getSolution1")
    public String getSolution() {
        System.out.println(1);
        return "getSolution1";
    }
}
