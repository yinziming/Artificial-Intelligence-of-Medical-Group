package com.example.strokereasoning.domain;

public class ResultData {
    private String[] arraypt;
    private String[] arrayot;
    private String[] arrayst;

    public ResultData(String[] arraypt, String[] arrayot, String[] arrayst) {
        this.arraypt = arraypt;
        this.arrayot = arrayot;
        this.arrayst = arrayst;
    }
    public String[] getStrPt() {
        return arraypt;
    }
    public String[] getStrOt() {
        return arrayot;
    }
    public String[] getStrSt() {
        return arrayst;
    }
}
//    public ResultData() {}




