(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun relu (x)
  (if (> x 0.0) x 0.0))

(defun softmax (x)
  (let* ((e (loop for i in x collect (exp i)))
		 (sum (reduce #'+ e)))
	(loop for i in e collect (/ i sum))))

(defun 1-log (x) (log (1- x)))

(defun cross-entropy (a y)
  (let ((n (length a)))
	(* (/ -1 n)
	   (reduce #'+
			   (mapcar #'+
					   (mapcar #'* y (mapcar #'log a))
					   (mapcar #'* (mapcar #'1- y) (mapcar #'1-log a)))))))

;; * (cross-entropy '(0.1 0.1) '(0.1 0.1))
;; #C(0.13543403 2.8274333)
