const functions = require('firebase-functions');
const admin = require('firebase-admin');
const nodemailer = require('nodemailer');

admin.initializeApp();

// Replace with your email and app password (not your Gmail login password)
const transporter = nodemailer.createTransport({
  service: 'gmail',
  auth: {
    user: 'yourapp@gmail.com',      // your Gmail
    pass: 'yourapppassword',        // App password, not regular password
});

exports.sendRejectionEmail = functions.firestore
  .document('Users/{userId}')
  .onUpdate((change, context) => {
    const before = change.before.data();
    const after = change.after.data();

    if (before.status !== 'rejected' && after.status === 'rejected') {
      const mailOptions = {
        from: 'yourapp@gmail.com',
        to: after.email,
        subject: 'Your Receipt Was Rejected',
        text: `Hi ${after.username || 'User'},\n\nYour receipt was rejected for the following reason:\n\n${after.rejection_reason}\n\nPlease re-upload a valid receipt.`,
      };

      return transporter.sendMail(mailOptions)
        .then(() => console.log('Email sent to:', after.email))
        .catch(error => console.error('Error sending email:', error));
    }

    return null;
  });
