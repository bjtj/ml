;; https://namu.wiki/w/표준%20편차
(defun stdev (seq)
  (let ((m (/ (reduce #'+ seq) (length seq))))
    (sqrt (/ (reduce #'+
		     (map 'list #'(lambda (x)
				    (expt (- x m) 2)) seq))
	     (length seq)))))
