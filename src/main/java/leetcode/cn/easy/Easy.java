package leetcode.cn.easy;

import java.util.Arrays;

/*
 * Created by yu_wei on 2019/3/13.
 *
 */
class ListNode {
    int val;
    ListNode next;

    public ListNode(int x) {
        this.val = x;
    }
}

public class Easy {

    public static void main(String[] args) {
        ListNode first = new ListNode(1);
        first.next = new ListNode(2);
        first.next.next = new ListNode(3);
        first.next.next.next = new ListNode(4);
        first.next.next.next.next = new ListNode(5);

        showLink(first);
        System.out.println("========================>");
        showLink(reverseLink(first));
    }

    /**
     * Num007
     * 给出一个 32 位的有符号整数，你需要将这个整数中每位上的数字进行反转。
     *
     * @param x
     * @return
     */
    public static int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7)) return 0;
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

    /**
     * Num008
     * 请你来实现一个 atoi 函数，使其能将字符串转换成整数。
     * 首先，该函数会根据需要丢弃无用的开头空格字符，直到寻找到第一个非空格的字符为止。
     * 当我们寻找到的第一个非空字符为正或者负号时，则将该符号与之后面尽可能多的连续数字组合起来，作为该整数的正负号；假如第一个非空字符是数字，则直接将其与之后连续的数字字符组合起来，形成整数。
     * 该字符串除了有效的整数部分之后也可能会存在多余的字符，这些字符可以被忽略，它们对于函数不应该造成影响。
     * 注意：假如该字符串中的第一个非空格字符不是一个有效整数字符、字符串为空或字符串仅包含空白字符时，则你的函数不需要进行转换。
     * 在任何情况下，若函数不能进行有效的转换时，请返回 0。
     * 说明：
     * <p>
     * 假设我们的环境只能存储 32 位大小的有符号整数，那么其数值范围为 [−231,  231 − 1]。如果数值超过这个范围，qing返回  INT_MAX (231 − 1) 或 INT_MIN (−231) 。
     *
     * @param str
     * @return
     */
    public static int myAtoi(String str) {
        String[] res = str.split("");
        System.out.println(Arrays.toString(res));

        int num = 0;
        if (res[0] == "") {
            return 0;
        }
        return 0;
    }

    public static int reverseNum(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > Integer.MAX_VALUE / 10 || (rev == Integer.MAX_VALUE / 10 && pop > 7)) return 0;
            if (rev < Integer.MIN_VALUE / 10 || (rev == Integer.MIN_VALUE / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }

    /**
     * Num206
     * 反转一个单链表。
     * 输入: 1->2->3->4->5->NULL
     * 输出: 5->4->3->2->1->NULL
     *
     * @param head
     * @return
     */
    public static ListNode reverseList(ListNode head) {
        ListNode dummyHead = new ListNode(0);
        while (head != null) {
            ListNode p = head;
            head = head.next;
            p.next = dummyHead.next;
            dummyHead.next = p;
        }
        return dummyHead.next;
    }

    public static ListNode reverseL(ListNode head) {
        // 1. 第一个条件是判断递归开始，传入的参数的合法性。第二个是递归的终止条件
        if (head == null || head.next == null) {
            return head;
        }
        // 2. 开始进行递归
        ListNode newHead = reverseL(head.next);
        // 3. 尾部4-5-null中，head=4,head.next=4-5 head.next.next=4-5-null,也就是5的后继指向4
        head.next.next = head;
        // 4. 断开之前4-5之间的连接，将4的后继指向null
        head.next = null;
        // 5. 返回已经反转的链表
        return newHead;
    }

    /**
     * Num206
     * 反转一个单链表。
     *
     * @param head
     * @return
     */
    public static ListNode reverseLink(ListNode head) {
        ListNode prev = null; // 前指针节点
        ListNode curr = head; // 当前指针节点
        // 每次循环，都将当前节点指向它前面的节点，然后当前节点和前节点后移
        while (curr != null) {
            ListNode nextTemp = curr.next; // 临时节点，暂存当前节点的下一节点，用于后移
            curr.next = prev; // 将当前节点指向它前面的节点
            prev = curr; // 前指针后移
            curr = nextTemp; // 当前指针后移
        }
        return prev;
    }

    public static void showLink(ListNode head) {
        ListNode first = head;
        while (first != null) {
            System.out.print(first.val + "\t");
            first = first.next;
        }
    }

}

class Temp {
        /*
     public static int reverse(int x) {
     int[] xList = new int[String.valueOf(x).length()];

     int res = 0;
     int i = 0;

     if (x >= 0) {
     while ((x / 10) > 0) {
     xList[i] = x % 10;
     x = x / 10;
     i++;
     }
     xList[i] = x;
     for (int j = i; j >= 0; j--) {
     res = (int) (xList[j] * Math.pow(10, i - j) + res);
     }
     } else {
     x = x * -1;
     while ((x / 10) > 0) {
     xList[i] = x % 10;
     x = x / 10;
     i++;
     }
     xList[i] = x;
     for (int j = i; j >= 0; j--) {
     res = (int) (xList[j] * Math.pow(10, i - j) + res);
     }
     res = res * -1;
     }

     return res;
     }*/
}
