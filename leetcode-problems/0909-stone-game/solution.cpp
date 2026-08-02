class Solution {
public:
    bool stoneGame(vector<int>& piles) {
        int n = piles.size();
        vector<vector<int>> dp(n, vector<int>(n, 0));
        for(int i = 0; i<n; i++){
            dp[i][i] = piles[i];
        }
        for(int i = 2; i<= n; i++){
            for(int j = 0; j<= n- i; i++){
                int k = j + i - 1;
                dp[j][k] = max(piles[j] - dp[j + 1][k],
                piles[k] - dp[j][k - 1]);
            }
        }
        return dp[0][n-1] > 0;
    }
};
