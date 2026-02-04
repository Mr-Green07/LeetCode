class Solution {
public:
    int minimumOperations(vector<int>& nums, int start, int goal) {
        queue<int> q;
        vector<bool> visited(1001, false);
        q.push(start);
        visited[start] = true;
        int steps = 0;
        while(!q.empty()){
            int size = q.size();
            while(size--){
                int curr = q.front();
                q.pop();
                if(curr == goal){
                    return steps;
                }
                for(int num: nums){
                    int nextvals[3] = {curr + num, curr- num, curr ^ num};
                    for(int next: nextvals){
                        if(next == goal){
                            return steps + 1;
                        }
                        if(next >= 0 && next <= 1000 && !visited[next]){
                            visited[next] = true;
                            q.push(next);
                        }
                    }
                }
            }
            steps++;
        }
        return -1;
    }
};
