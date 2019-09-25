/**
 * 排序算法练习
 */
public class Train_two {
    public static void main(String[] args) {
        System.out.println("Hello World!");
    }

    //数组两个数交换位置
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    //快速排序算法 https://blog.csdn.net/pengzonglu7292/article/details/
    //快速排序：递归
    public static void quickSort(int[] arr,int left, int right){

        if (left>right)
            return;
        int index=partition(arr,left,right);

        quickSort(arr,left,index-1);
        quickSort(arr,index+1,right);

    }
    //快速排序：找出分割点
    public static int partition(int[] arr, int left, int right){
        int pivot=arr[left];
        int i=left,j=right;
        while(i!=j){
            while (arr[j]>=pivot && i<j)
                j--;
            while (arr[i]<=pivot && i<j)
                i++;
            swap(arr,i,j);
        }
        swap(arr,left,i);

        return i;
    }
}
