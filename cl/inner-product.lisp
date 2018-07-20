;; http://www.iqool.de/A_Zillion_Ways.html


(defun IP (a b)
  (if (null a)
      0
      (+ (* (car a) (car b))
	 (IP (cdr a) (cdr b)))))

(defun IP (a b &optional (acc 0))
  (if (or a b)
      (IP (cdr a) (cdr b) (+ acc (* (car a) (car b))))
      acc))
