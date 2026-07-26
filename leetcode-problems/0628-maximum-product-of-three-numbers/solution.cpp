class Solution {
public:
    int maximumProduct(vector<int>& nums) {
        ranges::sort(nums);
        int n = nums.size();
        return max(
            nums[n-1] * nums[n-2] * nums[n-3],   // three largest
            nums[0]   * nums[1]   * nums[n-1]    // two smallest * largest
        );
    }
};
