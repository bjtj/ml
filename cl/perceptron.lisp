(defparameter *learning-rate* 0.1)
(defparameter *max-data-set* 500)
(defparameter *max-iteration* 100)

(defun activate (wx wy wb x y)
  (if (<= 0 (+ (* x wx) (* y wy) wb)) 1.0 -1.0))

;; (defparameter *train-data* '((0 0 -1) (0 1 1) (1 0 1) (1 1 1)))

(defun train (data)
  (defparameter *wx* (random 1.0))
  (defparameter *wy* (random 1.0))
  (defparameter *wb* (random 1.0))
  (format t "[TRAIN] wx: ~a wy: ~,4f wb: ~,4f~%" *wx* *wy* *wb*)
  (loop for i below *max-iteration*
	 do (let ((global-err 0))
		  (loop for (x y z) in data
			 do (let* ((predict (activate *wx* *wy* *wb* x y))
					   (err (- z predict)))
					(setf *wx* (+ *wx* (* *learning-rate* err x))
						  *wy* (+ *wy* (* *learning-rate* err y))
						  *wb* (+ *wb* (* *learning-rate* err))
						  global-err (+ global-err (* err err)))
					(format t "[~,4f:~4,f] wx: ~a wy: ~,4f wb: ~,4f~%" predict z *wx* *wy* *wb*)))
		  (format t "Iteration: ~a / RMSE: ~,4f / GE(Global Error): ~,4f~%"
				  i (sqrt (/ global-err (length data))) global-err)
		  (when (= global-err 0) (progn (format t "[DONE]") (return)))))
  (format t "x: ~,4f, y: ~,4f, b: ~,4f~%" *wx* *wy* *wb*)
  (loop for (x y z) in data
	   do (let* ((predict (activate *wx* *wy* *wb* x y))
					   (err (- z predict)))
			(format t "wx: ~a wy: ~,4f wb: ~,4f ==> ~a [~a]~%"
					*wx* *wy* *wb* predict (if (= predict z) "CORRECT" "WRONG"))))
  )
