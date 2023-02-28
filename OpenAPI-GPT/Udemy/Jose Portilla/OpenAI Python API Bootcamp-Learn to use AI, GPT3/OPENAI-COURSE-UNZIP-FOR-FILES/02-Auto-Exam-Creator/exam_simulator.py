import datetime

from teacher import Teacher

class Exam:
    def __init__(self, student_view, answers, store_test=False, topic=""):
        self.student_view = student_view
        self.answers = answers

        if store_test:
            self.store_test(topic)

    
    def take(self):
        answers = {}
        for question, question_view in self.student_view.items():
            print(question_view)
            answer = input("Enter your answer: ")
            answers[question] = answer
        return answers

    def grade(self, answers):
        correct_answers = 0
        for question, answer in answers.items():
            if answer.upper() == self.answers[question].upper()[16]:
                correct_answers+=1
        grade = 100 * correct_answers / len(answers)

        if grade < 60:
            passed = "Not passed!"
        else:
            passed = "Passed!"
        return f"{correct_answers} out of {len(answers)} correct! You achieved: {grade} % : {passed}"


    def store_test(self, topic):
        with open(f'Test_{topic}_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.txt', "w") as file:
            for question, question_view in self.student_view.items():
                file.write(question_view)
                file.write("\n")
                file.write(self.answers[question])
                file.write("\n")


if __name__ == "__main__":
    teacher = Teacher()
    student_view, answers = teacher.create_full_test()

    exam = Exam(student_view, answers, store_test=True, topic=teacher.test_creator.topic)
    student_answers = exam.take()
    print(student_answers)
    grade = exam.grade(student_answers)
    print(grade)
