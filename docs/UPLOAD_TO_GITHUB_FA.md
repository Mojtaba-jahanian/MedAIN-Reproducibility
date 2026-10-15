# بارگذاری مخزن MedAIN در GitHub

**مهم:** در GitHub صفحه `New repository` فایل ZIP را بارگذاری نکنید؛ ابتدا آن را Extract کنید. برای حفظ ساختار فولدرها، Git امن‌ترین مسیر است.

1. نام مخزن: `MedAIN-Reproducibility`؛ توضیح: `Reproducible analyses and archived results for MedAIN retinal segmentation under acquisition shift.`
2. پیشنهاد اولیه: `Private` تا حقوق انتشار، مقاله و داده‌ها کنترل شوند؛ در صورت تصمیم آگاهانه می‌توانید Public کنید.
3. `Add README` خاموش؛ `.gitignore` گزینه `No .gitignore`؛ License گزینه `No license` (تا تأیید تصمیم حقوقی).
4. مخزن را ایجاد کنید و آدرس HTTPS آن را در دستورهای زیر جایگزین کنید.

در PowerShell (پس از نصب Git و Extract All کردن بسته):

```powershell
cd "$HOME\Downloads\MedAIN_GitHub_Ready_V1"
python scripts/verify_results.py
git init
git add .
git commit -m "Initial MedAIN analysis and evidence release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MedAIN-Reproducibility.git
git push -u origin main
```

**نکته:** برای احراز هویت از مرورگر، Git Credential Manager یا GitHub CLI استفاده کنید؛ پسورد یا Token را داخل اسکریپت قرار ندهید.

در پایان، به Settings → Actions (در صورت فعال بودن) توجه کنید و موفقیت `Verify published aggregate calculations` را بررسی کنید.
