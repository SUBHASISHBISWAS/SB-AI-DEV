from teacher import Teacher
from exam_simulator import Exam

### Create an instance of the Teacher class which asks the user about the topic, number of possible answers per question, and number of questions.
teacher = Teacher()
student_view, answers = teacher.create_full_test()

### Create an instance of the Exam class which runs the exam simulation and grades the exam.
exam = Exam(student_view, answers, store_test=True, topic=teacher.test_creator.topic)
student_answers = exam.take()
print(student_answers)
grade = exam.grade(student_answers)
print(grade)
