import java.util.*;

class Solution {
    public int maximumUnits(int[][] boxTypes, int truckSize) {
        Arrays.sort(boxTypes, Comparator.comparingDouble(o -> o[1]));

        int maxUnits = 0;

        for (int i = boxTypes.length - 1; i >= 0; i--) { 
            if (truckSize >= boxTypes[i][0]) {
                maxUnits += (boxTypes[i][0] * boxTypes[i][1]);
                truckSize -= boxTypes[i][0];
            } else {
                maxUnits += truckSize * boxTypes[i][1];
                truckSize = 0;
                break;
            }
        }

        return maxUnits;
    }
}

// public class Main {
//     public static void main(String[] args) {
//         Scanner sc = new Scanner(System.in);

//         System.out.print("Enter number of box types: ");
//         int n = sc.nextInt();

//         int[][] boxTypes = new int[n][2];

//         System.out.println("Enter boxTypes (numberOfBoxes unitsPerBox):");
//         for (int i = 0; i < n; i++) {
//             boxTypes[i][0] = sc.nextInt();
//             boxTypes[i][1] = sc.nextInt();
//         }

//         System.out.print("Enter truck size: ");
//         int truckSize = sc.nextInt();

//         Solution sol = new Solution();
//         int result = sol.maximumUnits(boxTypes, truckSize);

//         System.out.println("Maximum units that can be loaded = " + result);
//     }
// }
