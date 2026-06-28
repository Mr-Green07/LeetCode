class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        vector<int> ans = nums1;
        ans.insert(ans.end(), nums2.begin(), nums2.end());
        sort(ans.begin(), ans.end());
        int n = ans.size();
        for(auto i : ans){
            cout<<i<<", ";
        }
        cout<<endl;
        if (n % 2 == 1)
            return ans[n/2];

        return (ans[n/2 - 1] + ans[n/2]) / 2.0;
    }
};

// use merge sort: nums1 = L, nums2 = r
// divide it in equal half
// insert the element from nums1 to l and nums2 to r

// compare elements and merge into main arr and then  find middle element and 
