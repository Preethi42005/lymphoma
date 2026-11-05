# LymphoScan - AI Lymphoma Subtype Detection

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Open your project in the [Lovable editor](https://lovable.dev) and keep building.

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: connect the project to GitHub and every change made in Lovable is committed straight to your repository.
- **Full ownership**: this code is yours. Push to your repository and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone https://github.com/Preethi42005/lymphoma.git
cd lymphoma
npm i
npm run dev
```

### Auth + Excel user storage

Sign-up and sign-in details are saved in `users.xlsx` (sheets: **Users**, **LoginHistory**).

1. Start the auth API (port 5002):

```sh
npm run auth
```

2. In another terminal, start the web app:

```sh
npm run dev
```

3. Open `/login` to create an account or sign in. New rows appear in `users.xlsx`.

Passwords are stored as SHA-256 hashes (not plain text). Close and reopen the Excel file after sign-up/login to see updates.

## Features

- **Hybrid Deep Learning Model**: MobileNetV2 + DenseNet121 with multi-scale feature fusion
- **Multi-scale Feature Extraction**: Features from different scales for better classification
- **Attention-based Fusion**: Custom attention mechanism for feature combination
- **Web Interface**: React 19 + TypeScript frontend with TanStack Router
- **Authentication System**: User login/signup with Excel storage
- **Exploratory Data Analysis**: Comprehensive dataset analysis and visualization
- **Performance Metrics**: Training curves, confusion matrix, and per-class metrics
- **QR Code Demo**: Mobile-friendly access to dashboard

## Classification Classes

- **MCL** (Mantle Cell Lymphoma)
- **FL** (Follicular Lymphoma)
- **CLL** (Chronic Lymphocytic Leukemia)

## Built with

- TanStack Start
- TypeScript
- React
- Tailwind CSS
- Flask auth API + pandas/openpyxl (`users.xlsx`)
- TensorFlow/Keras (model training)
