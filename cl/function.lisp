(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun relu (x)
  (if (> x 0.0) x 0.0))

(defun softmax (x)
  (let* ((e (loop for i in x collect (exp i)))
		(sum (reduce #'+ e)))
	(loop for i in e collect (/ i sum))))
