class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        const int INF = 1e9;
        vector<int> dp(amount + 1, INF);

        dp[0] = 0;
        for(int num: coins){
            for (int j = num; j<= amount; j++){
                dp[j] = min(dp[j], dp[j - num] + 1);
            }
        }
        return dp[amount] >= INF ? -1: dp[amount];
        
    }
};
