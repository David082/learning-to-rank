package leetcode;

import java.util.Stack;

/*
 * Created by yu_wei on 2019/2/22.
 */
public class StackPushPop {

    public static void main(String[] args) {
        Solution s = new Solution();
        s.push(0);
        s.pop();
    }

}

class Solution {
    private Stack<Integer> stack1 = new Stack<Integer>();
    private Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {
        stack1.push(node);
    }

    public int pop() {
        int a = 0;
        if (stack2.isEmpty()) {
            while (!stack1.isEmpty()) {
                a = stack1.pop();
                stack2.push(a);
            }
        }
        a = stack2.pop();
        return a;
    }
}
