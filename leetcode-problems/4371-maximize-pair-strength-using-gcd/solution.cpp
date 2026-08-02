class Solution {
public:
    long long maxPairStrength(vector<int>& nums) {
        long long ans = 0;
        int n = nums.size();
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                int g = gcd(nums[i], nums[j]);
                long long x = nums[i] / g;
                long long y = nums[j] / g;
                ans = max(ans, x * y);
            }
        }
        return ans;
    }
};
