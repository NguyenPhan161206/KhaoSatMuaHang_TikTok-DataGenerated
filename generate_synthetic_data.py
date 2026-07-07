import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, chi2_contingency
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

CSV_PATH = "Khảo sát__Tác động của các kích thích trong Livestream Commerce trên TikTok Shop đến sự hối hận sau mua và quyết định hoàn đơn_ Vai trò trung gian của hành vi mua bốc đồng ở người tiêu dùng thế hệ Z t.csv"
N_SYNTHETIC = 400
SEED = 42
np.random.seed(SEED)

# ============================================================
# 1. LOAD & CLEAN
# ============================================================
print("=" * 80)
print("1. LOAD & CLEAN DATA")
print("=" * 80)

df = pd.read_csv(CSV_PATH)
df['Dấu thời gian'] = pd.to_datetime(df['Dấu thời gian'], format='%d/%m/%Y %H:%M:%S')
df = df.replace(r'^\s*$', np.nan, regex=True)

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

categorical_cols_core = ['Purchased_via_Livestream', 'Living_in_HCMC_Area',
                         'Livestream_Frequency', 'Purchase_Frequency_Monthly',
                         'Most_Purchased_Product_Group', 'Return_Refund_Frequency']

categorical_cols_demo = ['Gender', 'Occupation', 'Monthly_Income', 'Birth_Year_Range']

numeric_cols = [col for col in df.columns if col not in
                ['Timestamp'] + categorical_cols_core + categorical_cols_demo]

print(f"  Rows: {len(df)} | Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols_core + categorical_cols_demo)}")

# ============================================================
# 2. SEPARATE COMPLETE CASES
# ============================================================
print("\n" + "=" * 80)
print("2. COMPLETE CASES (seed for generation)")
print("=" * 80)

complete = df.dropna(subset=numeric_cols).copy()
print(f"  Complete cases with full Likert data: {len(complete)}")
print(f"  Incomplete (NaN in Likert): {len(df) - len(complete)}")

# ============================================================
# 3. GAUSSIAN COPULA FOR NUMERIC LIKERT COLUMNS
# ============================================================
print("\n" + "=" * 80)
print("3. GAUSSIAN COPULA — GENERATE SYNTHETIC NUMERIC DATA")
print("=" * 80)

X = complete[numeric_cols].values.astype(float)
n_original, p = X.shape
print(f"  Seed matrix: {n_original}x{p}")

# Step 3a: Empirical CDF -> uniform (with jitter for discrete Likert)
def empirical_cdf_uniform(X):
    n, p = X.shape
    U = np.zeros_like(X, dtype=float)
    for j in range(p):
        col = X[:, j]
        unique_vals = np.sort(np.unique(col))
        for v in unique_vals:
            mask = (col == v)
            k = np.sum(col < v)
            m = np.sum(col == v)
            U[mask, j] = np.random.uniform(k / n, (k + m) / n, size=m)
    return U

U = empirical_cdf_uniform(X)
Z = norm.ppf(np.clip(U, 1e-10, 1 - 1e-10))

# Step 3b: Correlation matrix from normal-transformed data
corr_matrix = np.corrcoef(Z.T)
print(f"  Mean |correlation| (off-diag): {np.mean(np.abs(corr_matrix - np.eye(p))):.4f}")

eigvals = np.linalg.eigvalsh(corr_matrix)
if eigvals[0] < 1e-6:
    reg = abs(eigvals[0]) + 1e-4
    print(f"  Regularizing correlation matrix (+{reg:.4f}I)")
    corr_matrix += np.eye(p) * reg

# Step 3c: Generate MVN samples
Z_synth = np.random.multivariate_normal(np.zeros(p), corr_matrix, size=N_SYNTHETIC)
U_synth = norm.cdf(Z_synth)

# Step 3d: Inverse transform -> discrete Likert values
def inverse_empirical_discrete(U_synth, X_orig):
    n_syn = U_synth.shape[0]
    X_syn = np.zeros_like(U_synth, dtype=float)
    for j in range(U_synth.shape[1]):
        col = X_orig[:, j]
        unique_vals = np.sort(np.unique(col))
        n = len(col)
        for v in unique_vals:
            prob_v = np.mean(col == v)
        # Vectorized inverse CDF
        edges = np.cumsum([np.mean(col == v) for v in unique_vals])
        edges = np.clip(edges, 0, 1)
        for i in range(n_syn):
            idx = np.searchsorted(edges, U_synth[i, j])
            idx = min(idx, len(unique_vals) - 1)
            X_syn[i, j] = unique_vals[idx]
    return X_syn

X_numeric_synth = inverse_empirical_discrete(U_synth, X)

# Verify
mu_orig = X.mean(axis=0)
mu_synth = X_numeric_synth.mean(axis=0)
std_orig = X.std(axis=0)
std_synth = X_numeric_synth.std(axis=0)
print(f"  Mean abs diff in means: {np.abs(mu_orig - mu_synth).mean():.4f}")
print(f"  Mean abs diff in stds:  {np.abs(std_orig - std_synth).mean():.4f}")
print(f"  Value range: [{X_numeric_synth.min():.0f}, {X_numeric_synth.max():.0f}]")

# ============================================================
# 4. GENERATE CATEGORICAL DATA (Conditional Bootstrap)
# ============================================================
print("\n" + "=" * 80)
print("4. GENERATE CATEGORICAL DATA")
print("=" * 80)

cat_all = categorical_cols_core + categorical_cols_demo
purchase_cols = ['Living_in_HCMC_Area', 'Livestream_Frequency',
                  'Purchase_Frequency_Monthly', 'Most_Purchased_Product_Group',
                  'Return_Refund_Frequency']

# Use ORIGINAL 66 rows to determine proportion of Purchased_via_Livestream
p_yes = df['Purchased_via_Livestream'].value_counts(normalize=True).get('Có', 45/58)
n_yes = int(round(N_SYNTHETIC * p_yes))
n_no = N_SYNTHETIC - n_yes
print(f"  Target ratio: Yes={n_yes}, No={n_no} (from original {p_yes*100:.0f}% Yes)")

# Generate "Có" synthetic rows (with full data)
complete_yes = complete[complete['Purchased_via_Livestream'] == 'Có']
yes_idx = np.random.choice(len(complete_yes), size=n_yes, replace=True)
cat_yes = complete_yes.iloc[yes_idx][cat_all].reset_index(drop=True)

# Generate "Chưa" synthetic rows (demographics only, purchase cols = NaN)
df_no = df[df['Purchased_via_Livestream'] == 'Chưa'].copy()
if len(df_no) > 0 and n_no > 0:
    no_idx = np.random.choice(len(df_no), size=n_no, replace=True)
    cat_no = df_no.iloc[no_idx][cat_all].reset_index(drop=True)
    cat_no.loc[:, purchase_cols] = np.nan
else:
    cat_no = pd.DataFrame(columns=cat_all)

# Merge and shuffle
cat_synth = pd.concat([cat_yes, cat_no], ignore_index=True)
shuffled_idx = np.random.permutation(len(cat_synth))
cat_synth = cat_synth.iloc[shuffled_idx].reset_index(drop=True)

print(f"  Generated {len(cat_synth)} categorical rows")
vc = cat_synth['Purchased_via_Livestream'].value_counts(dropna=False)
print(f"  Purchased: Yes={vc.get('Có',0)}, No={vc.get('Chưa',0)}")

# ============================================================
# 5. TIMESTAMPS
# ============================================================
t_min = df['Timestamp'].min()
t_max = df['Timestamp'].max() + pd.Timedelta(days=30)
t_range_sec = (t_max - t_min).total_seconds()
synth_ts = t_min + pd.to_timedelta(np.random.uniform(0, t_range_sec, N_SYNTHETIC), unit='s')

# ============================================================
# 6. ASSEMBLE FINAL DATASET
# ============================================================
print("\n" + "=" * 80)
print("6. ASSEMBLE FINAL DATASET")
print("=" * 80)

# Build synthetic rows one by one to align numeric/categorical correctly
synth_rows = []
yes_counter = 0
no_counter = 0
for i in range(N_SYNTHETIC):
    row = {'Timestamp': synth_ts.iloc[i] if hasattr(synth_ts, 'iloc') else synth_ts[i]}
    is_yes = (cat_synth.iloc[i]['Purchased_via_Livestream'] == 'Có')
    for col in cat_all:
        row[col] = cat_synth.iloc[i][col]
    if is_yes:
        for col_idx, col in enumerate(numeric_cols):
            row[col] = X_numeric_synth[yes_counter, col_idx]
        yes_counter += 1
    else:
        for col in numeric_cols:
            row[col] = np.nan
    row['is_synthetic'] = 1
    synth_rows.append(row)

synth_df = pd.DataFrame(synth_rows, columns=['Timestamp'] + cat_all + numeric_cols + ['is_synthetic'])
assert yes_counter == n_yes, f"Expected {n_yes} Yes rows, got {yes_counter}"
assert no_counter == 0, "No-counter should be 0"

orig_df = df.copy()
orig_df['is_synthetic'] = 0

final_df = pd.concat([orig_df, synth_df], ignore_index=True)
col_order = ['Timestamp'] + cat_all + numeric_cols + ['is_synthetic']
final_df = final_df[col_order]

out_csv = "/home/binhnguyen/Downloads/DataGeneratedByBinhNguyen/survey_66_plus_400_synthetic.csv"
final_df.to_csv(out_csv, index=False)
print(f"  Saved: {out_csv}")
print(f"  Total: {len(final_df)} rows (66 original + {N_SYNTHETIC} synthetic)")
print(f"  Columns: {len(final_df.columns)}")

# Verify alignment
synth_yes = synth_df[synth_df['Purchased_via_Livestream'] == 'Có']
synth_no = synth_df[synth_df['Purchased_via_Livestream'] == 'Chưa']
print(f"  Synthetic Yes: {len(synth_yes)} rows, all numeric non-null: {synth_yes[numeric_cols].notna().all().all()}")
print(f"  Synthetic No:  {len(synth_no)} rows, all numeric null:  {synth_no[numeric_cols].isna().all().all()}")

# ============================================================
# 7. VALIDATION
# ============================================================
print("\n" + "=" * 80)
print("7. VALIDATION")
print("=" * 80)

Xn = complete[numeric_cols].values.astype(float)
Xs = X_numeric_synth

# 7a. KS test
print("\n--- KS Test (numeric columns) ---")
ks_rows = []
for i, col in enumerate(numeric_cols):
    stat, pval = stats.ks_2samp(Xs[:, i], Xn[:, i])
    ks_rows.append({'col': col[:40], 'KS_stat': f'{stat:.4f}', 'p_value': pval})
ks_df = pd.DataFrame(ks_rows)
ks_pass = (ks_df['p_value'] > 0.05).sum()
print(f"  Passed (p>0.05): {ks_pass}/{len(numeric_cols)} | Mean KS: {pd.to_numeric(ks_df['KS_stat']).mean():.4f}")
failed = ks_df[ks_df['p_value'] <= 0.05]
if len(failed) > 0:
    print(f"  FAILED (p<=0.05): {len(failed)} columns")
    for _, r in failed.iterrows():
        print(f"    {r['col']}: stat={r['KS_stat']}, p={r['p_value']:.4f}")

# 7b. Correlation difference
print("\n--- Correlation Comparison ---")
orig_corr = np.corrcoef(Xn.T)
synth_corr = np.corrcoef(Xs.T)
corr_diff = np.abs(orig_corr - synth_corr)
print(f"  Mean abs diff (all pairs): {corr_diff.mean():.4f}")
print(f"  Max abs diff:              {corr_diff.max():.4f}")

# Off-diagonal only
off_diag = ~np.eye(p, dtype=bool)
print(f"  Mean abs diff (off-diag):  {corr_diff[off_diag].mean():.4f}")

# 7c. Summary comparison
print("\n--- Summary Statistics ---")
summ = pd.DataFrame({
    'Column': numeric_cols,
    'Orig_Mean': Xn.mean(axis=0).round(2),
    'Syn_Mean': Xs.mean(axis=0).round(2),
    'Orig_Std': Xn.std(axis=0).round(2),
    'Syn_Std': Xs.std(axis=0).round(2),
    'Orig_Med': np.median(Xn, axis=0).round(1),
    'Syn_Med': np.median(Xs, axis=0).round(1),
})
summ['|Mean_Diff|'] = abs(summ['Orig_Mean'] - summ['Syn_Mean'])
summ['|Std_Diff|'] = abs(summ['Orig_Std'] - summ['Syn_Std'])
print(f"  Mean of |Mean_Diff|: {summ['|Mean_Diff|'].mean():.4f}")
print(f"  Max  of |Mean_Diff|: {summ['|Mean_Diff|'].max():.4f}")
print(f"  Mean of |Std_Diff|:  {summ['|Std_Diff|'].mean():.4f}")
print(f"  Max  of |Std_Diff|:  {summ['|Std_Diff|'].max():.4f}")

print("\n  Top 5 columns with largest mean diff:")
worst = summ.nlargest(5, '|Mean_Diff|')
for _, r in worst.iterrows():
    print(f"    {r['Column'][:35]}: orig={r['Orig_Mean']}, syn={r['Syn_Mean']}, diff={r['|Mean_Diff|']:.3f}")

# ============================================================
# 8. VISUALIZATION
# ============================================================
print("\n" + "=" * 80)
print("8. VISUALIZATIONS")
print("=" * 80)

out_dir = "/home/binhnguyen/Downloads/DataGeneratedByBinhNguyen"

# 8a. Density overlay (first 9 Likert columns)
n_plot = min(9, len(numeric_cols))
fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes_flat = axes.flatten()
for i in range(n_plot):
    ax = axes_flat[i]
    bins = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
    ax.hist(Xn[:, i], bins=bins, density=True, alpha=0.5, color='#2166AC',
            label=f'Original (n={n_original})', align='mid')
    ax.hist(Xs[:, i], bins=bins, density=True, alpha=0.4, color='#D6604D',
            label=f'Synthetic (n={N_SYNTHETIC})', align='mid')
    ax.set_title(numeric_cols[i][:40], fontsize=9)
    ax.set_xlim(0.5, 5.5)
    ax.legend(fontsize=7)
    ax.set_ylabel('Density')
fig.suptitle('Distribution Comparison: Original (39) vs Synthetic (400)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f'{out_dir}/validation_density_overlay.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: validation_density_overlay.png")

# 8b. Correlation heatmap
fig, axes = plt.subplots(1, 2, figsize=(22, 9))
for idx, (mat, lbl) in enumerate([
    (orig_corr, f'Original ({n_original} samples)'),
    (synth_corr, f'Synthetic ({N_SYNTHETIC} samples)')
]):
    ax = axes[idx]
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap='RdBu_r', aspect='equal')
    ax.set_title(f'Correlation Matrix — {lbl}', fontsize=11)
    ax.tick_params(axis='both', labelsize=4)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel('Column index'); ax.set_ylabel('Column index')
fig.suptitle('Correlation Structure: Original vs Synthetic', fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig(f'{out_dir}/validation_correlation_matrix.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: validation_correlation_matrix.png")

# 8c. Categorical comparison
cat_plot = ['Purchased_via_Livestream', 'Gender', 'Occupation', 'Monthly_Income',
            'Livestream_Frequency', 'Return_Refund_Frequency']
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes_flat = axes.flatten()
for i, col in enumerate(cat_plot):
    ax = axes_flat[i]
    oc = complete[col].value_counts(normalize=True, dropna=False)
    sc = cat_synth[col].value_counts(normalize=True, dropna=False)
    cats = sorted(set(list(oc.index) + list(sc.index)), key=lambda x: str(x))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, [oc.get(c, 0) for c in cats], w, label='Original',
           color='#2166AC', alpha=0.7)
    ax.bar(x + w/2, [sc.get(c, 0) for c in cats], w, label='Synthetic',
           color='#D6604D', alpha=0.7)
    ax.set_title(col, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c)[:18] for c in cats], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Proportion')
    ax.legend(fontsize=8)
fig.suptitle('Categorical Distribution: Original vs Synthetic', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig(f'{out_dir}/validation_categorical_comparison.png', dpi=120, bbox_inches='tight')
plt.close()
print("  Saved: validation_categorical_comparison.png")

# ============================================================
print("\n" + "=" * 80)
print("COMPLETE — ALL DONE!")
print("=" * 80)
print(f"  Data:      {out_csv}")
print(f"  Plots:     validation_density_overlay.png")
print(f"             validation_correlation_matrix.png")
print(f"             validation_categorical_comparison.png")
