class Solution {
public:
    int maxCoins(vector<int>& nums) {
        int n = nums.size();
        vector<int> a(n + 2);
        a[0] = 1;
        a[n + 1] = 1;
        for (int i = 0; i < n; i++) a[i + 1] = nums[i];
        int m = n + 2;
        vector<vector<int>> dp(m, vector<int>(m, 0));
        for (int len = 2; len < m; len++) {
            for (int left = 0; left + len < m; left++) {
                int right = left + len;
                for (int k = left + 1; k < right; k++) {
                    dp[left][right] = max(dp[left][right],
                        dp[left][k] + dp[k][right] + a[left] * a[k] * a[right]);
                }
            }
        }
        return dp[0][m - 1];
    }
};

