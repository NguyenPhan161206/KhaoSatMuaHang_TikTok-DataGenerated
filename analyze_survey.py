import pandas as pd
import numpy as np

CSV_PATH = "Khảo sát__Tác động của các kích thích trong Livestream Commerce trên TikTok Shop đến sự hối hận sau mua và quyết định hoàn đơn_ Vai trò trung gian của hành vi mua bốc đồng ở người tiêu dùng thế hệ Z t.csv"

df = pd.read_csv(CSV_PATH)

print("=" * 80)
print("1. THÔNG TIN DỮ LIỆU GỐC")
print("=" * 80)
print(f"Shape: {df.shape}")
print(f"Số dòng: {len(df)}")
print(f"Số cột: {len(df.columns)}")
print(f"\n5 dòng đầu:")
print(df.head().to_string(max_colwidth=30))
print(f"\ndf.info():")
df.info()

print("\n" + "=" * 80)
print("2. CHUYỂN TIMESTAMP -> DATETIME & THAY '' BẰNG NaN")
print("=" * 80)
df['Dấu thời gian'] = pd.to_datetime(df['Dấu thời gian'], format='%d/%m/%Y %H:%M:%S')
df = df.replace(r'^\s*$', np.nan, regex=True)

duplicate_rows = df.duplicated().sum()
print(f"Số dòng trùng lặp: {duplicate_rows}")
df.drop_duplicates(inplace=True)
print(f"Số dòng sau khi bỏ trùng: {len(df)}")

print("\n" + "=" * 80)
print("3. ĐỔI TÊN CỘT SANG TIẾNG ANH")
print("=" * 80)
new_column_names = {
    'Dấu thời gian': 'Timestamp',
    'Câu 1: Bạn sinh vào năm nào? ': 'Birth_Year_Range',
    'Câu 2: Bạn đã từng mua sản phẩm trên TikTok Shop thông qua livestream bán hàng chưa?': 'Purchased_via_Livestream',
    'Câu 3: Bạn đang sinh sống, học tập và làm việc tại khu vực Thành phố Hồ Chí Minh.': 'Living_in_HCMC_Area',
    'Câu 4: Tần suất bạn xem livestream bán hàng trên TikTok Shop là bao nhiêu?': 'Livestream_Frequency',
    'Câu 5: Trung bình trong 1 tháng, bạn mua hàng qua livestream trên TikTok Shop bao nhiêu lần?': 'Purchase_Frequency_Monthly',
    'Câu 6: Nhóm sản phẩm nào bạn THƯỜNG XUYÊN mua nhất qua livestream TikTok Shop? ': 'Most_Purchased_Product_Group',
    'Câu 7: Trong 6 tháng qua, bạn đã thực hiện yêu cầu "Trả hàng/Hoàn tiền" trên TikTok Shop bao nhiêu lần đối với các sản phẩm mua qua livestream?': 'Return_Refund_Frequency',
    'Câu 8: Thông tin về sản phẩm trong livestream rõ ràng và dễ hiểu.': 'Info_Clear_Easy_Understand',
    'Câu 9: Livestream cung cấp đầy đủ thông tin cần thiết giúp tôi dễ dàng đưa ra quyết định.': 'Livestream_Info_Sufficient',
    'Câu 10: Thông tin về sản phẩm và các chương trình ưu đãi trong livestream có độ chính xác cao.': 'Info_Accurate_Promotions',
    'Câu 11: Livestream trên TikTok Shop cung cấp những thông tin hữu ích và giúp tôi cập nhật xu hướng mới về sản phẩm. ': 'Livestream_Info_Useful_Trends',
    'Câu 12: Streamer trên TikTok Shop thường xuyên tiếp nhận phản hồi từ người xem.': 'Streamer_Receives_Feedback',
    'Câu 13: Người livestream khuyến khích sự giao tiếp và tương tác hai chiều với người xem.': 'Livestreamer_Encourages_Interaction',
    'Câu 14: Streamer thể hiện sự quan tâm đến ý kiến của người xem trong livestream.': 'Streamer_Cares_About_Viewer_Opinions',
    'Câu 15: Người xem có cơ hội trao đổi và tương tác với nhau trong livestream TikTok Shop.': 'Viewers_Interact_With_Each_Other',
    'Câu 16: Livestream trên TikTok Shop giúp tôi hiểu rõ hơn về công dụng của sản phẩm.': 'Livestream_Helps_Understand_Product_Use',
    'Câu 17: Những nội dung trình bày trong livestream giúp tôi đánh giá chất lượng sản phẩm hiệu quả hơn.': 'Livestream_Content_Helps_Evaluate_Quality',
    'Câu 18: Việc xem livestream giúp tôi có thêm cơ sở để đánh giá sản phẩm.': 'Watching_Livestream_Aids_Product_Evaluation',
    'Câu 19: Livestream giúp tôi có đủ thông tin để quyết định mình có muốn mua sản phẩm hay không.': 'Livestream_Provides_Enough_Info_For_Purchase_Decision',
    'Câu 20: KOL/KOC hiểu rất rõ về sản phẩm mà họ đang giới thiệu.': 'KOL_KOC_Understands_Product',
    'Câu 21: KOL/KOC có đủ kiến thức và kinh nghiệm để giải đáp các thắc mắc về sản phẩm.': 'KOL_KOC_Knowledgeable_Experienced',
    'Câu 22: Tôi đánh giá KOL/KOC là người có chuyên môn trong lĩnh vực sản phẩm họ đang livestream.': 'KOL_KOC_Professional_Expert',
    'Câu 23: KOL/KOC đưa ra những nhận xét chuyên sâu và hữu ích về sản phẩm.': 'KOL_KOC_Insightful_Helpful_Reviews',
    'Câu 24: KOL/KOC giới thiệu sản phẩm một cách trung thực': 'KOL_KOC_Introduces_Product_Honestly',
    'Câu 25: KOL/KOC chia sẻ trải nghiệm về sản phẩm một cách chân thành.': 'KOL_KOC_Shares_Experience_Sincerely',
    'Câu 26: Tôi cảm thấy KOL/KOC này là một nguồn thông tin đáng tin cậy.': 'KOL_KOC_Trustworthy_Source',
    'Câu 27: KOL/KOC luôn review sản phẩm một cách nghiêm túc và tận tâm.': 'KOL_KOC_Reviews_Seriously_Diligently',
    'Câu 28: Khi xem livestream trên TikTok Shop, tôi hoàn toàn tập trung vào nội dung.': 'Focused_On_Livestream_Content',
    'Câu 29: Xem livestream kích thích sự tò mò và khám phá của tôi.': 'Livestream_Stimulates_Curiosity_Exploration',
    'Câu 30: Bản thân việc xem livestream là điều thú vị đối với tôi.': 'Livestream_Watching_Is_Fun',
    'Câu 31: Khi xem livestream, tôi cảm thấy hoàn toàn bị cuốn vào nhịp điệu của buổi live mà không bị phân tâm. ': 'Engrossed_In_Livestream_Without_Distraction',
    'Câu 32: Tôi đã mua một sản phẩm vốn không nằm trong dự định ban đầu của mình.': 'Impulse_Purchase_Not_Planned',
    'Câu 33:  Quyết định mua sản phẩm của tôi bị ảnh hưởng bởi cảm xúc tại thời điểm đó.': 'Purchase_Decision_Influenced_By_Emotion',
    'Câu 34: Hành vi "thấy là mua" mô tả khá chính xác cách tôi mua hàng trong livestream.': 'See_Buy_Behavior',
    'Câu 35: Tôi thường mua sản phẩm một cách không có kế hoạch trước.': 'Frequent_Unplanned_Purchases',
    'Câu 36: Tôi ước mình đã dành thời gian tìm hiểu kỹ thông tin hơn trước khi bấm nút mua hàng.': 'Wished_Researched_More_Before_Purchase',
    'Câu 37: Tôi nghĩ mình đã quá vội vàng chốt đơn mà chưa kịp cân nhắc kỹ lưỡng.': 'Too_Hasty_To_Confirm_Order',
    'Câu 38: Nếu lý trí và chậm lại một chút, tôi tin rằng mình đã đưa ra quyết định tốt hơn.': 'Better_Decision_With_Rationality',
    'Câu 39: Tôi cảm thấy hối hận vì đã không suy nghĩ chín chắn trước khi quyết định mua sản phẩm này..': 'Regret_Not_Thinking_Carefully_Before_Purchase',
    'Câu 40: Tôi lẽ ra nên chọn một sản phẩm khác.  .': 'Should_Have_Chosen_Different_Product',
    'Câu 41: Tôi hối hận về quyết định mua hàng của mình.': 'Regret_Purchase_Decision',
    'Câu 42: Sau khi nhận được sản phẩm, tôi cảm thấy không hài lòng với quyết định mua hàng của mình.': 'Dissatisfied_After_Receiving_Product',
    'Câu 43: Nhìn lại, tôi ước gì mình đã không chi tiền để mua sản phẩm này.': 'Wished_Had_Not_Spent_Money',
    'Câu 44: Tôi cân nhắc việc hoàn trả sản phẩm đã mua (nhưng có thể chưa bấm nút thực hiện ngay).': 'Considering_Return_Product',
    'Câu 45:  Có khả năng cao tôi sẽ tiến hành hoàn trả sản phẩm đã mua này trên TikTok Shop.': 'Likely_To_Return_Product',
    'Câu 46: Tôi quyết định bấm nút yêu cầu "Trả hàng / Hoàn tiền" trên ứng dụng đối với sản phẩm này.': 'Decided_To_Request_Return_Refund',
    'Câu 47:  Việc hoàn trả sản phẩm là một lựa chọn mà tôi đang xem xét.': 'Product_Return_Option_Considered',
    'Câu 48: Nếu điều kiện cho phép, tôi sẽ tiến hành hoàn trả sản phẩm.': 'Will_Return_If_Conditions_Allow',
    'Câu 49: Tôi sẽ để lại đánh giá tiêu cực về sản phẩm.': 'Leave_Negative_Review_Product',
    'Câu 50: Tôi sẽ không giới thiệu sản phẩm này cho người khác.': 'Will_Not_Recommend_Product_Negative',
    'Câu 51: Tôi sẽ khuyên bạn bè và người thân không nên mua sản phẩm này.': 'Advise_Friends_Family_Not_Buy_Product',
    'Câu 52: Tôi sẽ chia sẻ trải nghiệm tiêu cực của mình về sản phẩm trên mạng xã hội.': 'Share_Negative_Experience_Social_Media',
    'Câu 53: Tôi sẽ để lại bình luận tiêu cực về sản phẩm.': 'Leave_Negative_Comment_Product',
    'Câu 53: Tôi sẽ tẩy chay đối với các bài đăng liên quan đến sản phẩm.': 'Boycott_Product_Posts',
    'Câu 54: Tôi sẽ chủ động chia sẻ hoặc tìm kiếm thêm các bài viết phản ánh tiêu cực về sản phẩm. ': 'Actively_Share_Seek_Negative_Reviews',
    'Câu 55: Tôi sẽ chia sẻ các nội dung tiêu cực liên quan đến trải nghiệm mua hàng của mình.': 'Share_Negative_Shopping_Content',
    'Câu 56: Giới tính của bạn': 'Gender',
    'Câu 57: Tình trạng công việc hiện tại của bạn là': 'Occupation',
    'Câu 58: Thu nhập hoặc chi tiêu trung bình hàng tháng của bạn là bao nhiêu?': 'Monthly_Income',
}
df = df.rename(columns=new_column_names)

print(f"Shape sau rename: {df.shape}")
print(f"\n5 dòng đầu (đã rename):")
print(df.head().to_string(max_colwidth=30))

print("\n" + "=" * 80)
print("4. KIỂM TRA KIỂU DỮ LIỆU (SAU RENAME)")
print("=" * 80)
df.info()

print("\n" + "=" * 80)
print("5. THỐNG KÊ GIÁ TRỊ DUY NHẤT - CỘT PHÂN LOẠI")
print("=" * 80)
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    print(f"\n--- {col} ---")
    print(df[col].value_counts(dropna=False))

print("\n" + "=" * 80)
print("6. THỐNG KÊ MÔ TẢ - CỘT SỐ (THANG ĐO 1-5)")
print("=" * 80)
numeric_cols = df.select_dtypes(include=[np.number]).columns
print(df[numeric_cols].describe().round(2))

print("\n" + "=" * 80)
print("==> DONE <==")
print("=" * 80)
