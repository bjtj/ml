(defun sigmoid (x)
  (/ 1 (+ 1 (exp (- x)))))

(defun relu (x)
  (if (> x 0.0) x 0.0))
