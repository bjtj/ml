(defun factorial (n)
  (if (= n 0)
	  1
	(* n (factorial (1- n)))))

(defun main ()
  (loop for i from 0 to 16
		do (format t "~D! = ~D~%" i (factorial i))))
