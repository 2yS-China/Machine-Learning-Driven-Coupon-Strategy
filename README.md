The project titled "Machine Learning-Driven Coupon Strategy" aims to find the optimal coupon allocation strategy. First, data preprocessing and analysis are performed using data_visualization.py. The data is cleaned and the following features are derived:

User Dimensions: User's spending amount, average order amount, purchase frequency, number of orders
Merchant Dimensions: Merchant order volume, sales, and shipping cost ratio
RFM Analysis: Recency (recent consumption behavior), Frequency (purchase frequency), Monetary (spending amount)
Coupon Usage Record: Whether the coupon was used and the usage rate
Next, the code behavior_model.py is run, where I used Random Forest, XGBoost, and Deep Neural Networks (DNN) for prediction.

Random Forest: Learns based on features whether a user will use a coupon. The result is a binary prediction (0 or 1—whether the coupon will be used or not).
XGBoost: Compared to Random Forest, XGBoost is better at handling non-linear relationships, improving prediction performance. The result is the prediction of whether a user will use the coupon, with metrics such as accuracy and F1 score.
Deep Neural Networks (DNN): Uses deep learning to discover complex patterns in coupon usage. The result is the final prediction of whether a user will use the coupon.
Finally, after the model is trained, a probability prediction and binning strategy are applied for optimization, leading to the development of the allocation strategy.

