package leetcode.datastructure;

import java.util.Arrays;

/*
 * Created by yu_wei on 2019/3/4.
 *
 * https://github.com/iTimeTraveler/SortAlgorithms/tree/master/src/main/java/com/example
 */
public class Sort {
    public static void main(String[] args) {
        int[] arr = new int[]{5, 3, 9, 1, 6, 4, 10, 2, 8, 7, 15, 3, 2};
        System.out.println("=============================> bubble sort");
        bubbleSort(arr);
        System.out.println("=============================> quick sort");
        quickSort(arr, 0, arr.length - 1);
    }

    /**
     * 冒泡排序
     * ①. 比较相邻的元素。如果第一个比第二个大，就交换他们两个。
     * ②. 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
     * ③. 针对所有的元素重复以上的步骤，除了最后一个。
     * ④. 持续每次对越来越少的元素重复上面的步骤①~③，直到没有任何一对数字需要比较。
     *
     * @param arr 待排序数组
     */
    public static int[] bubbleSort(int[] arr) {
        for (int i = 0; i < arr.length - 1; i++) {
            for (int j = 0; j < arr.length - 1 - i; j++) {
                if (arr[j] > arr[j + 1]) {
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                }
            }
            System.out.println(Arrays.toString(arr));
        }
        return arr;
    }

    /**
     * 快速排序（递归）
     * ①. 从数列中挑出一个元素，称为"基准"（pivot）。
     * ②. 重新排序数列，所有比基准值小的元素摆放在基准前面，所有比基准值大的元素摆在基准后面（相同的数可以到任一边）。在这个分区结束之后，该基准就处于数列的中间位置。这个称为分区（partition）操作。
     * ③. 递归地（recursively）把小于基准值元素的子数列和大于基准值元素的子数列排序。
     *
     * @param arr   待排序数组
     * @param start 左边界
     * @param end   右边界
     */
    public static int[] quickSort(int[] arr, int start, int end) {
        int standard = arr[start];
        int low = start;
        int high = end;

        while (low < high) {
            while (low < high && standard < arr[high]) {
                high--;
            }
            arr[low] = arr[high];
            while (low < high && arr[low] < standard) {
                low++;
            }
            arr[high] = arr[low];
        }
        arr[low] = standard;

        return arr;
    }

}
