class Solution {
public:
    vector<int> findErrorNums(vector<int>& nums) {
        int duplicate = -1;
        int missing = -1;
        for(int i = 0; i< nums.size(); i++){
            int idx = abs(nums[i]) - 1;
            if(nums[idx] <0){
                duplicate = abs(nums[i]);
            }else{
                nums[idx] = -nums[idx];
            }
        }
        for (int i = 0; i<nums.size(); i++){
            if (nums[i]>0){
                missing = i+1;
            }
        }
        return {duplicate, missing};
    }
};
