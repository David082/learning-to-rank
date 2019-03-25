package leetcode;

/*
 * Created by yu_wei on 2019/2/26.
 */
public class FindNumsAppearOnce {
    public static void main(String[] args) {

    }

    public void findNumsAppearOne(int[] array, int num1[], int num2[]) {
        if (array == null || array.length < 2) {
            return;
        }
        int temp = 0;
        for (int i = 0; i < array.length; i++) {
            temp ^= array[i];

            int indexOf1;
        }
    }

    public int findFirstBilts(int num) {
        int indexBit = 0;
        while (((num & 1) == 0) && (indexBit) < 8 * 4) {
            num = num >> 1;
            ++indexBit;
        }
        return indexBit;
    }

    public boolean isBit(int num, int indexBit) {
        num = num >> indexBit;
        return (num & 1) == 1;
    }
}
