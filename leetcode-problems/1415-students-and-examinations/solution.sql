SELECT
    s.student_id,
    s.student_name,
    su.subject_name,
    COUNT(e.student_id) AS attended_exams
FROM 
    students s
cross join
    subjects su
left JOIN 
    examinations e ON s.student_id = e.student_id
    and su.subject_name = e.subject_name
GROUP BY 
    s.student_id,
    s.student_name,
    su.subject_name
ORDER BY 
    s.student_id ASC,
    su.subject_name asc
