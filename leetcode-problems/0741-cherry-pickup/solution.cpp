class Solution {
public:
    int cherryPickup(vector<vector<int>>& grid) {
        int n = grid.size();
        const int NEG = -1e9;

        vector<vector<int>> dp(n, vector<int>(n, NEG));
        dp[0][0] = grid[0][0];

        for (int k = 1; k <= 2 * (n - 1); k++) {
            vector<vector<int>> ndp(n, vector<int>(n, NEG));

            for (int r1 = max(0, k - (n - 1)); r1 <= min(n - 1, k); r1++) {
                for (int r2 = max(0, k - (n - 1)); r2 <= min(n - 1, k); r2++) {
                    int c1 = k - r1;
                    int c2 = k - r2;

                    if (grid[r1][c1] == -1 || grid[r2][c2] == -1) continue;

                    int best = NEG;

                    // both came from up
                    best = max(best, dp[r1][r2]);

                    // first from left, second from up
                    if (r1 > 0) best = max(best, dp[r1 - 1][r2]);

                    // first from up, second from left
                    if (r2 > 0) best = max(best, dp[r1][r2 - 1]);

                    // both came from left
                    if (r1 > 0 && r2 > 0) best = max(best, dp[r1 - 1][r2 - 1]);

                    if (best < 0) continue;

                    int cherries = grid[r1][c1];
                    if (r1 != r2) cherries += grid[r2][c2];

                    ndp[r1][r2] = best + cherries;
                }
            }

            dp = move(ndp);
        }

        return max(0, dp[n - 1][n - 1]);
    }
};
