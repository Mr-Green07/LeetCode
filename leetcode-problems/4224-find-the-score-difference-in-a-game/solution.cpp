class Solution {
public:
    int scoreDifference(vector<int>& nums) {
        int s1 = 0; 
        int s2 = 0;
        bool act = true;
        for(int i = 0; i<  nums.size(); ++i){
            if(nums[i] % 2 !=0){
                act = !act;
            }
            if((i + 1) % 6 == 0){
                act = !act;
            }
            if(act){
                s1 += nums[i];
            }else{
                s2 += nums[i];
            }
        }
        return s1-s2;
    }
};
