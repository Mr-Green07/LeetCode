class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        
        if(len(nums1)> len(nums2)):
            return self.findMedianSortedArrays(nums2, nums1)
        
        m, n = len(nums1), len(nums2)
        low = 0
        high = m
        while(low<= high):
            cut = (low + high) //2
            cut1 = (m + n + 1) // 2 - cut
            l1 = float('-inf') if(cut == 0) else nums1[cut - 1]
            r1 = float('inf') if(cut == m) else nums1[cut]

            l2 = float('-inf') if(cut1 == 0) else nums2[cut1 - 1]
            r2 = float('inf') if(cut1 == n) else nums2[cut1]
            if(l1<= r2 and l2<=r1):
                if( (m+ n) % 2 ==0):
                    return (max(l1, l2) + min(r1, r2))/2
                else:
                    return max(l1, l2)
            elif(l1> r2):
                high = cut -1
            else:
                low = cut + 1

        
        
