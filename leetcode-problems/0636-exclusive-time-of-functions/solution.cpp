class Solution {
public:
    struct Log {
        int id;
        bool isStart;
        int time;
    };

    // Helper to parse the log string cleanly
    Log parse(const string& log) {
        int first = log.find(':');
        int second = log.find_last_of(':');
        int id = stoi(log.substr(0, first));
        bool isStart = log.substr(first + 1, second - first - 1) == "start";
        int time = stoi(log.substr(second + 1));
        return {id, isStart, time};
    }

    vector<int> exclusiveTime(int n, vector<string>& logs) {
        vector<int> res(n, 0);
        stack<int> st;
        int prev_time = 0;

        for (const string& logStr : logs) {
            Log log = parse(logStr);

            if (log.isStart) {
                if (!st.empty()) {
                    // Add time to the function currently at the top of the stack
                    res[st.top()] += log.time - prev_time;
                }
                st.push(log.id);
                prev_time = log.time;
            } else {
                // Function ending: calculate inclusive time (+1)
                res[st.top()] += log.time - prev_time + 1;
                st.pop();
                // Next function starts at the very next second
                prev_time = log.time + 1;
            }
        }
        return res;
    }
};
