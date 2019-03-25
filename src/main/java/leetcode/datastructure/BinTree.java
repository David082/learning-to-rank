package leetcode.datastructure;

/*
 * Created by yu_wei on 2019/3/5.
 */
public class BinTree {
    public static void main(String[] args) {
        BinaryTree binTree = new BinaryTree();
        TreeNode root = new TreeNode(1);
        binTree.setRoot(root);

        TreeNode rootLeftChild = new TreeNode(2);
        root.setLeftNode(rootLeftChild);
        TreeNode rootRightChild = new TreeNode(3);
        root.setRightNode(rootRightChild);

        rootLeftChild.setLeftNode(new TreeNode(4));
        rootLeftChild.setRightNode(new TreeNode(5));
        rootRightChild.setLeftNode(new TreeNode(6));
        rootRightChild.setRightNode(new TreeNode(7));

        System.out.println(root.leftNode.data);
        System.out.println("=================> front tree");
        binTree.frontShow();
        System.out.println();
        System.out.println("=================> mid tree");
        binTree.midShow();
        System.out.println();
        System.out.println("=================> after tree");
        binTree.afterShow();
    }
}

class BinaryTree {
    TreeNode root;

    public void setRoot(TreeNode root) {
        this.root = root;
    }

    public TreeNode getRoot() {
        return root;
    }

    public void frontShow() {
        this.root.frontShow();
    }

    public void midShow() {
        this.root.midShow();
    }

    public void afterShow() {
        this.root.afterShow();
    }
}

class TreeNode {
    int data;

    TreeNode leftNode;
    TreeNode rightNode;

    public TreeNode(int data) {
        this.data = data;
    }

    public void setLeftNode(TreeNode leftNode) {
        this.leftNode = leftNode;
    }

    public void setRightNode(TreeNode rightNode) {
        this.rightNode = rightNode;
    }

    public void frontShow() {
        System.out.print(this.data + "\t");
        if (this.leftNode != null) {
            leftNode.frontShow();
        }
        if (this.rightNode != null) {
            rightNode.frontShow();
        }
    }

    public void midShow() {
        if (this.leftNode != null) {
            leftNode.midShow();
        }
        System.out.print(this.data + "\t");
        if (this.rightNode != null) {
            rightNode.midShow();
        }
    }

    public void afterShow() {
        if (this.leftNode != null) {
            leftNode.afterShow();
        }
        if (this.rightNode != null) {
            rightNode.afterShow();
        }
        System.out.print(this.data + "\t");
    }
}
