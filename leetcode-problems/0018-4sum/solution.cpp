class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> res;
        int n = nums.size();
        if(n<4) return res;
        sort(nums.begin(), nums.end());
        for(int i = 0; i<n -3; ++i){
            if(i>0 && nums[i] == nums[i-1]) continue;
            for(int j = i + 1; j< n-2; ++j){
                if(j > i + 1 && nums[j] == nums[ j-1]) continue;
                int l = j + 1;
                int r = n - 1;
                while (l < r){
                    long long twosum = (long long)nums[i] + nums[j] + nums[l] + nums[r];
                    if(twosum ==  target){
                        res.push_back({nums[i], nums[j], nums[l], nums[r]});
                        l++;
                        r--;

                        int lastl = nums[l - 1], lastr = nums[r + 1];
                        while(l < r && nums[l] == lastl) l++;
                        while(l< r && nums[r] == lastr) r--;
                    }else if (twosum < target){
                        l++;
                    }else{
                        r--;
                    }
                }
            }
        } 
        return res;   
    }
};
