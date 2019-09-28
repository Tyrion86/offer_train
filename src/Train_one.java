/**
 * 《剑指offer》的练习
 */

import java.util.*;

public class Train_one {
    public static void main(String[] args) {

        TreeNode a=new TreeNode(10);
        TreeNode b=new TreeNode(5);
        TreeNode c=new TreeNode(4);
        TreeNode d=new TreeNode(7);
        TreeNode e=new TreeNode(12);
        a.left=b;a.right=e;
        b.left=c;b.right=d;
        ArrayList<ArrayList<Integer>> result=Test34(a,22);

    }


    //数组两个数交换位置
    public static void swap(int[] arr, int i, int j) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }
    public static void swap(char[] arr, int i, int j) {
        char temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
    }

    /**
     * 链表结构
     */
    public class ListNode {
        int val;
        ListNode next = null;

        ListNode(int val) {
            this.val = val;
        }
    }

    /**
     * Definition for binary tree
     */
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

    /**
     * 定义TreeLinkNode,树结构
     */
    public class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode parent = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }

    /**
     * 复杂链表节点
     */
    public class RandomListNode {
        int label;
        RandomListNode next = null;
        RandomListNode random = null;

        RandomListNode(int label) {
            this.label = label;
        }
    }
    //找出数组中重复数字(已知数组的数字范围是0~n-1，数组长度n)  TC：O(n) SC: O(1) 辅助空间O(n)
    public static boolean Test3_1(int[] a) {
        if (a == null || a.length <= 1)
            return false;
        for (int i = 0; i < a.length; i++) {
            if (a[i] < 0 || a[i] > a.length - 1)
                return false;
        }

        for (int i = 0; i < a.length; i++) {

            while (a[i] != i) {
                if (a[i] == a[a[i]]) {
                    //int resulet=a[i];
                    return true;
                } else {//交换
                    int temp = a[i];
                    a[i] = a[a[i]];
                    a[a[i]] = temp;
                }
            }
        }
        return false;
    }

    //找出数组重复数字，不能改变原数组结构，数字范围是1~n，数组长度n+1  TC：O(nlogn) SC：O(1)
    public static int Test3_2(int[] a) {
        if (a == null || a.length <= 1)
            return -1;

        int start = 1;
        int end = a.length - 1;
        while (start <= end) {
            int middle = ((end - start) >> 1) + start;
            int count = getCountRange(start, middle, a);
            if (start == end) {
                if (count > 1)
                    return start;
                else
                    break;
            }
            if (count > middle - start + 1)
                end = middle;
            else
                start = middle + 1;
        }
        return -1;
    }

    //Test3_2中使用的方法，返回数组中start至end范围的数字的个数
    public static int getCountRange(int start, int end, int[] a) {
        int count = 0;
        for (int i = 0; i < a.length; i++) {
            if (a[i] >= start && a[i] <= end)
                count++;
        }
        return count;
    }

    //二维数组每行左到右递增，每列上到下递增，查找是否包含特定数字
    public static boolean Test4(int[][] a, int num) {
        int row = 0;
        int column = a[0].length - 1;
        boolean result = false;

        if (a != null) {
            while (column >= 0 && row < a.length) {
                if (a[row][column] == num) {
                    result = true;
                    break;
                } else if (a[row][column] > num)
                    column--;
                else if (a[row][column] < num)
                    row++;
            }
        }
        return result;
    }

    //将字符串中的每个空格替换成 %20
    public static String Test5(StringBuffer a) {
        if (a == null)
            return null;

        int numOfSpace = 0;
        for (int i = 0; i < a.length(); i++) {
            if (a.charAt(i) == ' ')
                numOfSpace++;
        }
        int p1 = a.length() - 1;//原数组长度
        a.setLength(a.length() + numOfSpace * 2);
        int p2 = a.length() - 1;//新数组长度

        while (p1 >= 0) {
            if (a.charAt(p1) != ' ') {
                a.setCharAt(p2, a.charAt(p1));
                p2--;
            } else {
                a.replace(p2 - 2, p2 + 1, "%20");
                p2 -= 3;
            }
            p1--;
        }
        return a.toString();
    }

    //从尾至头打印链表(不改变原结构，非递归)
    public static ArrayList<Integer> Test6(ListNode listNode) {
        ArrayList<Integer> list = new ArrayList<>();
        ListNode temp = listNode;
        while (temp != null) {
            list.add(0, temp.val);//每次都添加到第一个，后面的会顺延，即形成一个栈结构
            temp = temp.next;
        }
        return list;

    }


    //输入二叉树前序遍历和中序遍历的结果，重现二叉树
    public TreeNode Test7(int[] pre, int[] in) {
        if (pre.length == 0 || in.length == 0 || pre.length != in.length)
            return null;
        TreeNode root = new TreeNode(pre[0]);
        for (int i = 0; i < in.length; i++) {
            if (pre[0] == in[i]) {
                //Arrays.copyOfRange复制成为新的数组，原数组，包含起始，不包含末尾
                root.left = Test7(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
                root.right = Test7(Arrays.copyOfRange(pre, i + 1, pre.length), Arrays.copyOfRange(in, i + 1, in.length));
                break;
            }
        }
        return root;
    }


    //找出中序遍历的下一个节点
    public TreeLinkNode Test8(TreeLinkNode pNode) {
        //该节点有右子树
        if (pNode.right != null) {
            TreeLinkNode temp = pNode.right;
            while (temp.left != null) {
                temp = temp.left;
            }
            return temp;
        }
        //该节点无右子树，且是父节点的左子树
        else if (pNode.parent != null && pNode.parent.left == pNode) {
            return pNode.parent;
        }
        //该节点无右子树,是父节点的右子树
        else if (pNode.parent != null) {
            TreeLinkNode temp = pNode.parent;
            while (temp.parent != null && temp.parent.right == temp) {
                temp = temp.parent;
            }
            return temp.parent;
        }
        return null;
    }


    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    //用两个栈实现队列
    public void Test9() {
        //在牛客网实现
    }

    //斐波那契数列的第n项
    public static int Test10(int n) {
        int[] result = {0, 1};
        if (n < 2)
            return result[n];

        for (int i = 1; i < n; i++) {
            int temp = result[1];
            result[1] = result[0] + temp;
            result[0] = temp;
        }
        return result[1];
    }

    //旋转数组中最小值（分为两种情况讨论，一种是递增的，一种是形如11011111的数组）
    public static int Test11(int[] arr) {
        int left = 0, right = arr.length - 1;

        if (right == 0)
            return 0;
        //第二种情况
        if (arr[(right - left) / 2] == arr[left] || arr[(right - left) / 2] == arr[right]) {
            int min = arr[0];
            for (int i = 1; i < arr.length; i++) {
                if (min > arr[i])
                    min = arr[i];
            }
            return min;

        }
        //第一种情况
        while (right - left != 1) {
            int mid = (left + right) / 2;
            if (arr[mid] >= arr[left]) {
                left = mid;
                continue;
            }
            if (arr[mid] <= arr[right]) {
                right = mid;
                continue;
            }
        }
        return arr[right];
    }

    //矩阵中是否存在路径   回溯法
    public static boolean Test12(char[] matrix, int rows, int cols, char[] str) {
        if (matrix == null || rows < 1 || cols < 1 || str == null)
            return false;
        boolean[] visited = new boolean[matrix.length];

        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < cols; col++) {
                if (hasPathCore(matrix, rows, cols, str, row, col, 0, visited))
                    return true;
            }
        }
        return false;
    }

    public static boolean hasPathCore(char[] matrix, int rows, int cols, char[] str, int row, int col, int pathLenth, boolean[] visited) {
        boolean hasPath = false;
        int index = row * cols + col;

        if (row >= 0 && row < rows && col >= 0 && col < cols && str[pathLenth] == matrix[index] && !visited[row * cols + col]) {
            pathLenth++;
            visited[index] = true;
            if (pathLenth == str.length)
                return true;

            hasPath = hasPathCore(matrix, rows, cols, str, row, col + 1, pathLenth, visited) ||
                    hasPathCore(matrix, rows, cols, str, row + 1, col, pathLenth, visited) ||
                    hasPathCore(matrix, rows, cols, str, row, col - 1, pathLenth, visited) ||
                    hasPathCore(matrix, rows, cols, str, row - 1, col, pathLenth, visited);

            if (!hasPath) {
                visited[index] = false;
                pathLenth--;
            }

        }
        return hasPath;
    }

    //机器人运动范围   回溯法
    public static int Test13(int threadhold, int rows, int cols) {
        if (threadhold < 0 || rows < 1 || cols < 1)
            return 0;
        boolean[] visited = new boolean[rows * cols];
        int count = movingCountCore(threadhold, rows, cols, 0, 0, visited);
        return count;
    }

    public static int movingCountCore(int threadhold, int rows, int cols, int row, int col, boolean[] visited) {
        //检查格子是否满足要求
        int sum = 0;
        if (row >= 0 && row < rows && col >= 0 && col < cols && getDigitSum(row) + getDigitSum(col) <= threadhold && !visited[row * cols + col]) {
            visited[row * cols + col] = true;

            sum = 1 + movingCountCore(threadhold, rows, cols, row + 1, col, visited) +
                    movingCountCore(threadhold, rows, cols, row, col + 1, visited) +
                    movingCountCore(threadhold, rows, cols, row - 1, col, visited) +
                    movingCountCore(threadhold, rows, cols, row, col - 1, visited);
        }
        return sum;
    }

    //计算整数各位数字和
    public static int getDigitSum(int num) {
        int sum = 0;
        while (num > 0) {
            sum += num % 10;
            num /= 10;
        }
        return sum;
    }

    //剪绳子   动态规划与贪婪算法
    public static int Test14(int length) {
        if (length < 2)
            return 0;
        if (length == 2)
            return 1;
        if (length == 3)
            return 2;
        int[] products = new int[length + 1];

        products[0] = 0;
        products[1] = 1;
        products[2] = 2;
        products[3] = 3;

        int max = 0;

        for (int i = 4; i <= length; i++) {
            max = 0;
            for (int j = 1; j <= length / 2; j++) {
                int product = products[j] * products[i - j];
                if (product > max)
                    max = product;
                products[i] = max;
            }

        }
        max = products[length];
        return max;

    }

    //二进制中1的个数
    public static int Test15(int num) {
        int count = 0;
        while (num != 0) {
            count++;
            num = num & (num - 1);    //将num最右边的一变成0，并将计数器count++
        }
        return count;
    }

    //整数次方 保证base和exponent不同时为0
    public static double Test16(double base, int exponent) {
        if (exponent == 0)
            return 1;
        if (exponent == 1)
            return base;

        int ABexponent = exponent > 0 ? exponent : -exponent;
        double result = Test16(base, ABexponent / 2);
        result *= result;
        if ((exponent & 0x1) == 1)    //&0x1等同于%2，效率更高
            result *= base;

        result = exponent > 0 ? result : 1 / result;
        return result;

    }

    //打印从1到最大的n位数，例如，输入3，打印1-999    int->char方法：(char)(int+'0')
    public static void Test17(int n) {
        if (n < 1)
            return;

        char[] number = new char[n];
        for (int i = 0; i < 10; i++) {
            number[0] = (char) (i + '0');
            printToMax(number, n, 0);
        }
    }

    public static void printToMax(char[] number, int length, int index) {
        if (index == length - 1) {
            printNum(number);
            return;
        }
        for (int i = 0; i < 10; ++i) {
            number[index + 1] = (char) (i + '0');
            printToMax(number, length, index + 1);
        }

    }

    //打印数组中数字
    public static void printNum(char[] number) {
        boolean beginWith0 = true;
        for (int i = 0; i < number.length; i++) {
            if (number[i] != '0' && beginWith0)
                beginWith0 = false;
            if (!beginWith0)
                System.out.print(number[i]);
        }
        System.out.print(' ');
    }

    //删除链表节点
    public static void Test18_1(ListNode listHead, ListNode p) {
        if (listHead == null || p == null)
            return;
        //若删除结点不是尾节点
        if (p.next != null) {
            p.val = p.next.val;
            p.next = p.next.next;
        }
        //链表只有一个节点
        else if (p == listHead) {
            listHead = null;
        }
        //删除结点是尾结点
        else {
            ListNode temp = listHead;
            while (temp.next != p)
                temp = temp.next;
            temp.next = null;
            p = null;
        }

    }

    //在一个排序的链表中，删除链表重复节点
    public ListNode Test18_2(ListNode pHead) {
        if (pHead == null || pHead.next == null)
            return pHead;
        ListNode temp = new ListNode(Integer.MIN_VALUE);
        temp.next = pHead;

        ListNode cur = temp;
        ListNode pre = temp.next;

        while (cur != null) {
            if (cur.next != null && cur.next.val == cur.val) {
                while (cur.next != null && cur.next.val == cur.val)
                    cur = cur.next;
                cur = cur.next;
                pre.next = cur;
            } else {
                pre = cur;
                cur = cur.next;
            }
        }
        return temp.next;
    }

    //正则表达式匹配
    public static boolean Test19(char[] str, char[] pattern) {
        if (str == null || pattern == null)
            return false;
        return match(str, 0, pattern, 0);
    }

    public static boolean match(char[] str, int strIndex, char[] pattern, int patternIndex) {
        if (strIndex == str.length && patternIndex <= pattern.length)
            return true;
        if (strIndex != str.length && patternIndex == pattern.length)
            return false;

        //第二个字符是*,三种处理方式
        // 1.模式后移两位，相当于出现零次  2.字符串后移一位，模式后移两位，相当于出现一次 3.字符串后移一位，模式不变，相当于检查是否有继续后移  23两种情况可以合并
        if (patternIndex + 1 < pattern.length && pattern[patternIndex + 1] == '*') {
            if (strIndex != str.length && str[strIndex] == pattern[patternIndex] || strIndex != str.length && pattern[patternIndex] == '.') {
                return match(str, strIndex, pattern, patternIndex + 2) ||
                        match(str, strIndex + 1, pattern, patternIndex);
            } else
                return match(str, strIndex, pattern, patternIndex + 2);
        }
        //第二个字符不是*,且str和pattern匹配
        if (strIndex != str.length && pattern[patternIndex] == str[strIndex] || strIndex != str.length && pattern[patternIndex] == '.') {
            return match(str, strIndex + 1, pattern, patternIndex + 1);
        }

        return false;
    }

    //判断字符串是否表示数值 例如+100   -123   3.1456   -1E-6   .123
    //数字格式可以用以下标识  A[.[B]][e|EC] 或 .B[e|EC]
    //A是整数，B是小数部分，EC是指数部分，AC可以有正负号，但B不能有
    public static boolean Test20(char[] str) {
        //正则表达
        String s = String.valueOf(str);
        return s.matches("[+-]?\\d*(\\.\\d+)?([eE][+-]?\\d+)?");
        //书上解法参见https://www.nowcoder.com/profile/982154/codeBookDetail?submissionId=1513490
    }

    //调整数组顺序，使奇数在偶数前面   双指针，分别从前后扫描
    public static void Test21(int[] array) {
        if (array == null || array.length < 2)
            return;

        int left = 0, right = array.length - 1;
        while (left < right) {
            while (left < right && (array[left] & 0x1) == 1)
                left++;
            while (left < right && (array[right] & 0x1) == 0)
                right--;
            swap(array, left, right);
        }
    }

    //链表中的倒数第k个节点
    public static ListNode Test22(ListNode head, int k) {
        ListNode pre, q;
        pre = q = head;
        int i = 0;

        for (; pre != null; i++) {
            if (i >= k)
                q = q.next;
            pre = pre.next;
        }
        return i < k ? null : q;
    }

    //返回链表中环的入口节点
    public static ListNode Test23(ListNode head) {
        if (head == null || head.next == null)
            return null;
        ListNode index1, index2;
        index1 = index2 = head;
        while (index1 != null && index1.next != null) {
            index1 = index1.next.next;
            index2 = index2.next;
            if (index1 == index2) {
                index2 = head;
                while (index1 != index2) {
                    index1 = index1.next;
                    index2 = index2.next;
                }
                if (index1 == index2)
                    return index2;
            }
        }
        return null;
    }

    //反转链表
    public static ListNode Test24(ListNode head) {
        if (head == null || head.next == null)
            return head;

        ListNode node = head.next;
        ListNode pre = head;
        head.next = null;
        while (node.next != null) {
            ListNode temp = node.next;
            node.next = pre;

            pre = node;
            node = temp;
        }
        node.next = pre;
        return node;

    }

    //合并两个排序链表
    public static ListNode Test25(ListNode list1, ListNode list2) {
        if (list1 == null)
            return list2;
        if (list2 == null)
            return list1;

        ListNode result = null;
        if (list1.val < list2.val) {
            result = list1;
            result.next = Test25(list1.next, list2);
        } else {
            result = list2;
            result.next = Test25(list1, list2.next);
        }

        return result;
    }

    //树的子结构
    public boolean Test26(TreeNode root1, TreeNode root2) {
        if (root1 == null || root2 == null)
            return false;

        boolean result = false;
        if (root1.val == root2.val) {
            result = ifTreeSame(root1, root2);
        }
        if (!result) {
            result = Test26(root1.left, root2);
        }
        if (!result) {
            result = Test26(root1.right, root2);
        }
        return result;
    }

    //判断两棵树是否一样
    public static boolean ifTreeSame(TreeNode root1, TreeNode root2) {

        if (root2 == null)
            return true;
        if (root1 == null)
            return false;

        if (root1.val != root2.val)
            return false;


        return ifTreeSame(root1.left, root2.left) && ifTreeSame(root1.right, root2.right);
    }

    //二叉树镜像
    public static void Test27(TreeNode root) {
        if (root == null || root.left == null && root.right == null)
            return;

        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        Test27(root.left);
        Test27(root.right);
    }

    //二叉树是否对称
    public static boolean Test28(TreeNode root) {
        return isSymmetrical(root, root);
    }

    public static boolean isSymmetrical(TreeNode root1, TreeNode root2) {
        if (root1 == null && root2 == null)
            return true;
        if (root1 == null || root2 == null)
            return false;

        if (root1.val == root2.val)
            return isSymmetrical(root1.left, root2.right) && isSymmetrical(root1.right, root2.left);

        return false;
    }

    //顺时针打印矩阵
    public ArrayList<Integer> Test29(int[][] matrix) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return null;
        ArrayList<Integer> result = new ArrayList<>();
        int start = 0;

        while (start * 2 < matrix.length && start * 2 < matrix[0].length) {
            ArrayList<Integer> arr = printCircle(matrix, start);
            start++;
            for (int i = 0; i < arr.size(); i++) {
                result.add(arr.get(i));
            }
        }
        return result;
    }

    public static ArrayList<Integer> printCircle(int[][] matrix, int start) {
        int down = matrix.length - start - 1;
        int right = matrix[0].length - start - 1;
        ArrayList<Integer> result = new ArrayList<>();
        //左到右
        for (int i = start; i <= right; i++)
            result.add(matrix[start][i]);
        //上到下
        if (start < down - 1) {
            for (int i = start + 1; i < down; i++)
                result.add(matrix[i][right]);
        }
        //右到左
        if (start < down) {
            for (int i = right; i >= start; i--)
                result.add(matrix[down][i]);
        }
        //下到上
        if (start < down - 1 && start < right) {
            for (int i = down - 1; i > start; i--)
                result.add(matrix[i][start]);
        }
        return result;
    }

    //包含min函数的栈
    //在牛客网实现

    //栈的压入弹出序列
    public static boolean Test31(int [] pushA,int [] popA){
        if (pushA==null || popA==null || popA.length!=pushA.length)
            return false;
        Stack<Integer> stack=new Stack<Integer>();
        int j=0;
        for (int i=0;i<pushA.length;i++){
            stack.push(pushA[i]);

            while (!stack.empty() && stack.peek()==popA[j]){
                stack.pop();
                j++;
            }
        }
        return stack.empty();
    }

    //从上到下打印二叉树,每层从左到右打印
    public ArrayList<Integer> Test32(TreeNode root){
        ArrayList<Integer> arr=new ArrayList<Integer>();
        if (root==null)
            return arr;     //返回arr，不直接返回null

        Queue<TreeNode> que=new LinkedList<TreeNode>();
        que.add(root);
        while (!que.isEmpty()){
            TreeNode temp=que.poll();   //弹出首节点
            arr.add(temp.val);
            if (temp.left!=null)
                que.add(temp.left);
            if (temp.right!=null)
                que.add(temp.right);
        }
        return arr;
    }

    //按层输出二叉树，每一层输出一行
    public ArrayList< ArrayList<Integer> > Test32_1(TreeNode root){
        ArrayList<ArrayList<Integer>> arr=new ArrayList<ArrayList<Integer>>();
        if (root==null)
            return arr;

        int toBePrinted=1;
        int nextLevel=0;

        Queue<TreeNode> que=new LinkedList<TreeNode>();
        que.add(root);
        ArrayList<Integer> line=new ArrayList<Integer>();
        while (!que.isEmpty()){
            TreeNode temp=que.poll();
            if (toBePrinted>0){
                line.add(temp.val);
                toBePrinted--;
            }

            if (temp.left!=null){
                que.add(temp.left);
                nextLevel++;
            }
            if (temp.right!=null){
                que.add(temp.right);
                nextLevel++;
            }

            if (toBePrinted==0){
                arr.add(line);
                line=new ArrayList<Integer>();  //不能用line.clear，clear函数将所有对象赋值为null
                toBePrinted=nextLevel;
                nextLevel=0;
            }
        }
        return arr;
    }

    //之字形打印二叉树,第一行从左到右，第二行从右到左
    public ArrayList<ArrayList<Integer> > Test32_2(TreeNode root){
        ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
        if (root==null)
            return result;
        int level=1;

        Stack<TreeNode> odd=new Stack<TreeNode>();
        Stack<TreeNode> even=new Stack<TreeNode>();
//        ArrayList<Integer> line=new ArrayList<Integer>();
        odd.add(root);

        while (!odd.isEmpty() || !even.isEmpty()){
            if (!odd.isEmpty()){
                ArrayList<Integer> line=new ArrayList<Integer>();
                while (!odd.isEmpty()){
                    TreeNode temp=odd.pop();
                    line.add(temp.val);
                    if (temp.left!=null)
                        even.push(temp.left);
                    if (temp.right!=null)
                        even.push(temp.right);
                }
                result.add(line);
            }
            else {
                ArrayList<Integer> line=new ArrayList<Integer>();
                while (!even.isEmpty()){
                    TreeNode temp=even.pop();
                    line.add(temp.val);
                    if (temp.right!=null)   //注意顺序和奇数行操作不同
                        odd.push(temp.right);
                    if (temp.left!=null)
                        odd.push(temp.left);
                }
                result.add(line);
            }

        }
        return result;

    }

    //判断序列是否是二叉搜索树的后序遍历
    public boolean Test33(int [] sequence){
        if (sequence==null || sequence.length<1)
            return false;

        int root=sequence[sequence.length-1];
        int i=0;
        for (;i<sequence.length-1;i++){
            if (sequence[i]>root)
                break;
        }
        int j=i;
        for (;j<sequence.length-1;j++){
            if (sequence[j]<root)
                return false;
        }
        boolean left=true,right=true;
        if (i>0)
            left=Test33(Arrays.copyOfRange(sequence,0,i));
        if (j<sequence.length-1)
            right=Test33(Arrays.copyOfRange(sequence,i,sequence.length-1));
        return left&&right;
    }

    //二叉树路径和为某一值
    public static ArrayList<ArrayList<Integer>> Test34(TreeNode root,int target){
        ArrayList<ArrayList<Integer>> result=new ArrayList<ArrayList<Integer>>();
        if (root==null)
            return result;
        ArrayList<Integer> arr=new ArrayList<Integer>();
        find(root,target,result,arr,0);
        return result;
    }
    public static void find(TreeNode root, int target, ArrayList<ArrayList<Integer>> result, ArrayList<Integer> arr, int sum){
        if (root==null)
            return;
        sum+=root.val;

        if (root.left==null && root.right==null){
            if (sum==target){
                arr.add(root.val);
                result.add(new ArrayList<Integer>(arr));
                arr.remove(arr.size()-1);
            }
            return;
        }

        arr.add(root.val);
        find(root.left,target,result,arr,sum);
        find(root.right,target,result,arr,sum);
        arr.remove(arr.size()-1);
    }

    //复杂链表的复制
    public RandomListNode Test35(RandomListNode pHead){
        if (pHead==null)
            return null;
        //第一步：对于每个节点，复制一个到原节点后面
        RandomListNode node=pHead;
        while (node!=null){
            RandomListNode temp=new RandomListNode(node.label);
            temp.next=node.next;
            node.next=temp;
            node=temp.next;
        }
        //第二步：设置复制节点的random指针
        node=pHead;
        while (node!=null){
            RandomListNode cloned=node.next;
            cloned.random=node.random.next;
            node=cloned.next;
        }
        //第三步：拆分链表，奇数节点是原链表，偶数节点是新链表
        node=pHead;
        RandomListNode clonedHead=node.next;
        RandomListNode odd=node;
        RandomListNode even=node.next;

        while (node!=null){
            odd.next=node.next.next;
            even.next=even.next.next;
            node=node.next.next;
        }
        return clonedHead;

    }

    //二叉搜索树与双向链表
    public TreeNode Convert(TreeNode pRootOfTree){
        TreeNode root = pRootOfTree;
        if(root==null)
            return null;

        //定位左子树链表的头节点
        TreeNode left = Convert(root.left);
        TreeNode p = left;
        //定位左子树链表的尾节点
        while(p!=null&&p.right!=null)
            p = p.right;

        //3.如果左子树链表不为空，则将当前root节点追加到左子树上
        if(left!=null)
        {
            p.right = root;
            root.left = p;
        }
        //4.如果右子树链表不为空，则将该链表追加的root节点之后。
        TreeNode right= Convert(root.right);
        if(right!=null)
        {
            root.right = right;
            right.left = root;
        }

        //5.根据左子树链表是否为空，返回相应的头节点
        if(left!=null)
            return left;

        return root;
    }

    //序列化和反序列化函数
    public String Test36_1(TreeNode root){
        if (root==null)
            return "";
        StringBuilder sb=new StringBuilder();
        Serialize(root,sb);
        return sb.toString();

    }
    public void Serialize(TreeNode root, StringBuilder sb){
        if (root==null){
            sb.append("#!");
        }
        else {
            sb.append(root.val);
            sb.append("!");
            Serialize(root.left,sb);
            Serialize(root.right, sb);
        }
    }

    //字符串的排列
    public static ArrayList<String> Test38(String str){
        ArrayList<String> result=new ArrayList<String>();
        if (str==null || str.length()<1)
            return result;
        if (str.length()==1){
            result.add(str);
            return result;
        }
        char[] chars=str.toCharArray();

        return sort(chars,0,result);
    }
    public static ArrayList<String> sort(char[] chars, int index, ArrayList<String> result){
        if (index==chars.length-1 && !result.contains(String.valueOf(chars)))
            result.add(String.valueOf(chars));
        else {
            for (int i=index;i<chars.length-1;i++){
                swap(chars,index,i);
                sort(chars,index+1,result);
            }
        }
        return result;
    }
}
